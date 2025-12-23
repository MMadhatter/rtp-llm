#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include "rtp_llm/cpp/normal_engine/ContextParallelUtils.h"
#include <cstdlib>
#include <cstring>
#include <memory>
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/models/GptModel.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/models/NativeDeviceGraphModel.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"
#include <iostream>

using namespace std;

namespace rtp_llm {

NormalExecutor::NormalExecutor(const EngineInitParams&                   params,
                               const std::shared_ptr<CacheManager>&      cache_manager,
                               rtp_llm::DeviceBase*                      device,
                               const std::shared_ptr<lora::LoraManager>& lora_manager,
                               bool                                      warm_up):
    Executor(device),
    cache_manager_(cache_manager),
    lora_manager_(lora_manager),
    warm_up_(warm_up),
    use_all_gather_(params.gpt_init_parameter.use_all_gather_),
    metrics_reporter_(params.metrics_reporter),
    tps_reporter_(MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector>(metrics_reporter_)) {
    auto& gpt_param    = params.gpt_init_parameter;
    enable_detail_log_ = gpt_param.profiling_debug_logging_config.enable_detail_log;
    RTP_LLM_LOG_INFO("enable_detail_log_ = %d", enable_detail_log_);

    if (gpt_param.enable_eplb_ && gpt_param.moe_style_ != 0) {
        // use first moe layer weight as moe weight type
        int  first_moe_layer = gpt_param.moe_layer_index_.front();
        auto moe_weight_type = params.gpt_weights.layers[first_moe_layer].ffn_weights.moe_gate_weight->kernel->type();

        expert_balancer_ = make_shared<ExpertBalancer>(gpt_param.expert_num_,
                                                       gpt_param.phy_exp_num_,
                                                       gpt_param.num_layers_,
                                                       gpt_param.moe_inter_padding_size_,
                                                       gpt_param.hidden_size_,
                                                       gpt_param.eplb_update_time_,
                                                       gpt_param.ep_rank_,
                                                       gpt_param.ep_size_,
                                                       gpt_param.py_eplb_,
                                                       moe_weight_type,
                                                       device_,
                                                       gpt_param.eplb_mode_,
                                                       gpt_param.quant_algo_,
                                                       metrics_reporter_);
    }

    int               eos_id = params.gpt_init_parameter.special_tokens_.eos_token_id_;
    SamplerInitParams sampler_params{
        device_,
        eos_id,
        device->initParams().max_batch_size};  // set static max batch size to avoid sampler reset memory
    sampler_.reset(new Sampler(sampler_params));

    GptModelInitParams model_init_params(
        {device_,
         params.gpt_weights,
         genModelDescription(params.gpt_init_parameter),
         cache_manager ? ((optional<KVCacheAllocator::KVCacheBuffer>)cache_manager->kvCacheBuffer()) : nullopt,
         params.model_id});

    if (params.gpt_init_parameter.ffn_disaggregate_config.enable_ffn_disaggregate) {
        RTP_LLM_LOG_INFO("using ffn as service");
        enable_ffn_disaggregate_ = true;
    }
    if (!params.py_model.is_none()) {
        RTP_LLM_LOG_INFO("init executor with python model");
        model_.reset(new PyWrappedModel(model_init_params, params.py_model));
    } else if (device_->initParams().hw_kernel_config.enable_native_cuda_graph) {
        RTP_LLM_LOG_INFO("init legacy c++ gpt model with native cuda graph");
        model_.reset(new NativeDeviceGraphModel(model_init_params));
    } else {
        RTP_LLM_LOG_INFO("init legacy c++ gpt model");
        model_.reset(new GptModel(model_init_params));
    }

    // when warmup, cache manager maybe nullptr
    const auto& cache_config = cache_manager ? cache_manager->cacheConfig() : CacheConfig();
    batch_stream_processor_  = NormalBatchStreamProcessor::create(params.gpt_init_parameter, cache_config, warm_up_);
    PrefixToCandidateTokens::instance()->reloadPrefixDictWithPrefix(
        params.gpt_init_parameter.ckpt_path_, params.gpt_init_parameter.sp_config.tree_decode_config);
    device_->profileStart();
}

absl::Status NormalExecutor::process(const std::list<GenerateStreamPtr>& streams) {
    StreamGroups                   stream_groups(streams);
    RtpLLMExecutorMetricsCollector executor_collector;
    RtpLLMTokenPSMetricsCollector  tps_collector;
    GptModelInputs                 model_input;
    GptModelOutputs                model_output;
    SamplerOutput                  sampler_output;
    {
        int64_t start_time_us      = autil::TimeUtility::currentTimeInMicroSeconds();
        auto    model_input_status = batch_stream_processor_->gatherModelInput(stream_groups);
        RETURN_IF_STATUS_OR_ERROR(model_input_status);
        model_input                              = std::move(model_input_status.value());
        executor_collector.gather_model_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }
    {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        model_input.skip_run  = streams.empty() && !enable_ffn_disaggregate_;
        tpSyncModelInputs(model_input, device_);
        if (model_input.skip_run) {
            return absl::OkStatus();
        }

        if (device_->getDeviceProperties().cp_size > 0) {
            // handle context parallel inputs, do padding and shuffle and slice
            int cp_rank = device_->getDeviceProperties().cp_rank;
            int cp_size = device_->getDeviceProperties().cp_size;
            RTP_LLM_LOG_INFO("[cp_rank: %d], model_input: %s", cp_rank, model_input.debugString().c_str());
            std::cout << "[cp_rank: " << cp_rank << "], model_input: " << model_input.debugString().c_str() << endl;
            handleContextParallelInputs(model_input, cp_rank, cp_size);
            std::cout << "[cp_rank: " << cp_rank << "], model_input after slice: " << model_input.debugString().c_str()
                      << endl;
            RTP_LLM_LOG_INFO("[cp_rank: %d], model_input after slice: %s", cp_rank, model_input.debugString().c_str());
        }

        executor_collector.tp_sync_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }
    {
        // update kv cache
        if (model_input.kv_cache_update_mapping) {
            cache_manager_->blockBatchCopy(*model_input.kv_cache_update_mapping);
        }
    }
    // get lora input
    if (lora_manager_) {
        model_input.lora_model_input =
            lora_manager_->makeLoraModelInput(model_input.lora_ids, model_input.lora_input_lengths);
    }
    {
        bool force = device_->getDeviceProperties().tp_rank == 0 && enable_detail_log_;
        if (force) {
            RTP_LLM_LOG_INFO("model_input: %s", model_input.debugString(force).c_str());
        } else {
            RTP_LLM_LOG_TRACE("model_input: %s", model_input.debugString(force).c_str());
        }
        int64_t start_time_us               = autil::TimeUtility::currentTimeInMicroSeconds();
        model_output                        = std::move(model_->forward(model_input));
        executor_collector.model_forward_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
        RTP_LLM_LOG_DEBUG("model forward done");
    }
    if (expert_balancer_) {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        expert_balancer_->stepForward(*model_, executor_collector);
        executor_collector.eplb_step_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }

    if (device_->getDeviceProperties().tp_rank > 0 || device_->getDeviceProperties().cp_rank > 0 || warm_up_
        || streams.size() == 0) {
        return absl::OkStatus();
    }
    {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        CHECK_AND_RETURN_REF(sampler_input,
                             batch_stream_processor_->gatherSamplerInput(stream_groups, model_input, model_output));
        sampler_output = std::move(sampler_->forward(sampler_input));
        RTP_LLM_LOG_DEBUG("sampler forward done");
        executor_collector.sample_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }
    {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        auto    result =
            batch_stream_processor_->dispatch(stream_groups, {std::move(model_output), std::move(sampler_output)});
        executor_collector.dispatch_output_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
        reportMetrics(stream_groups, executor_collector, tps_collector);
        return result;
    }
}

void NormalExecutor::reportMetrics(const StreamGroups&             stream_groups,
                                   RtpLLMExecutorMetricsCollector& executor_collector,
                                   RtpLLMTokenPSMetricsCollector&  tps_collector) {
    if (device_->getDeviceProperties().tp_rank > 0 || device_->getDeviceProperties().cp_rank > 0) {
        return;
    }
    if (metrics_reporter_) {
        executor_collector.context_batch_size  = stream_groups.totalContextBatchSize();
        executor_collector.generate_batch_size = stream_groups.totalDecodeBatchSize();
        executor_collector.execute_token_size  = stream_groups.modelExecuteTokenSize();
        executor_collector.max_seq_len         = stream_groups.maxSeqLen();
        if (executor_collector.context_batch_size != 0) {
            executor_collector.context_batch_size_when_has_context  = executor_collector.context_batch_size;
            executor_collector.generate_batch_size_when_has_context = executor_collector.generate_batch_size;
            executor_collector.execute_token_size_when_has_context  = executor_collector.execute_token_size;
            executor_collector.max_seq_len_when_has_context         = executor_collector.max_seq_len;
        }
        metrics_reporter_->report<RtpLLMExecutorMetrics, RtpLLMExecutorMetricsCollector>(nullptr, &executor_collector);

        tps_collector.context_tps  = stream_groups.modelExecuteTokenSize() - stream_groups.totalDecodeBatchSize();
        tps_collector.generate_tps = stream_groups.totalDecodeBatchSize();
        tps_collector.total_tps    = stream_groups.modelExecuteTokenSize();
        tps_reporter_.report(&tps_collector);
    }
}

void NormalExecutor::handleContextParallelInputs(GptModelInputs& model_input, int cp_rank, int cp_size) {
    auto& total_input_tokens = model_input.combo_tokens;
    auto& input_lengths      = model_input.input_lengths;
    auto& sequence_lengths   = model_input.sequence_lengths;
    // auto& prefix_lengths = model_input.prefix_lengths;
    auto& prefill_cp_padding_lengths = model_input.prefill_cp_padding_lengths;
    auto& prefill_cp_chunk_lengths   = model_input.prefill_cp_chunk_lengths;
    auto& prefill_shuffle_indices    = model_input.prefill_shuffle_indices;

    int num_decode_stream  = sequence_lengths->shape()[0];
    int num_prefill_stream = input_lengths->shape()[0] - num_decode_stream;

    size_t cp_split_tokens_size = num_decode_stream;
    for (int p = 0; p < num_prefill_stream; ++p) {
        cp_split_tokens_size += static_cast<size_t>(prefill_cp_chunk_lengths->data<int>()[p]);
    }

    auto cp_split_input_tokens = CACHED_HOST_BUF(TYPE_INT32, {cp_split_tokens_size});

    int* input_token_ptr             = (int*)cp_split_input_tokens->data();
    int* input_length_ptr            = (int*)input_lengths->data();
    int* prefill_shuffle_indices_ptr = (int*)prefill_shuffle_indices->data();

    int input_token_idx       = 0;
    int total_input_token_idx = 0;
    // handle decode stream, directly memcpy input tokens
    if (num_decode_stream > 0) {
        std::memcpy(input_token_ptr,
                    total_input_tokens->dataWithOffset<int>(total_input_token_idx),
                    num_decode_stream * sizeof(int));
        input_token_idx += num_decode_stream;
        total_input_token_idx += num_decode_stream;
    }
    // handle prefill stream
    for (int p = 0; p < num_prefill_stream; ++p) {
        int input_chunk_length   = prefill_cp_chunk_lengths->data<int>()[p];
        int input_padding_length = prefill_cp_padding_lengths->data<int>()[p];
        int input_length         = input_lengths->data<int>()[p + num_decode_stream];

        // Copy input tokens for this prefill stream
        int*             src_tokens = total_input_tokens->dataWithOffset<int>(total_input_token_idx);
        std::vector<int> total_input_token_vec(src_tokens, src_tokens + input_length);
        std::vector<int> chunk_input_token(input_chunk_length, 0);
        std::vector<int> shuffle_index(input_chunk_length, -1);
        bool             success = contextParallelLoadBalanceSplit(total_input_token_vec,
                                                       chunk_input_token,
                                                       shuffle_index,
                                                       cp_rank,
                                                       cp_size,
                                                       input_chunk_length,
                                                       input_padding_length);
        RTP_LLM_CHECK_WITH_INFO(success, "contextParallelLoadBalanceSplit failed for prefill stream %d", p);
        // 写回新构造的input输入
        input_length_ptr[p + num_decode_stream] = input_chunk_length;
        std::memcpy(input_token_ptr + input_token_idx, chunk_input_token.data(), input_chunk_length * sizeof(int));
        std::memcpy(
            prefill_shuffle_indices_ptr + input_token_idx, shuffle_index.data(), input_chunk_length * sizeof(int));
        input_token_idx += input_chunk_length;
        total_input_token_idx += input_length;
    }
    // update model_input combo tokens
    model_input.combo_tokens = std::move(cp_split_input_tokens);
}
bool NormalExecutor::updateEplbConfig(const EplbConfig& config) {
    if (expert_balancer_) {
        return expert_balancer_->updateEplbConfig(config);
    }
    return true;
}

}  // namespace rtp_llm
