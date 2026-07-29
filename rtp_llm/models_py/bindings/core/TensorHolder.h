#pragma once

#include <cstddef>
#include <queue>
#include <vector>

#include <torch/extension.h>

namespace rtp_llm {

struct TensorHolderClearStats {
    size_t tensor_count        = 0;
    size_t pinned_tensor_count = 0;
    size_t pinned_bytes        = 0;

    TensorHolderClearStats& operator+=(const TensorHolderClearStats& other) {
        tensor_count += other.tensor_count;
        pinned_tensor_count += other.pinned_tensor_count;
        pinned_bytes += other.pinned_bytes;
        return *this;
    }
};

struct TensorHolder {
    static constexpr size_t kReleasedHoldRounds = 2;

    std::vector<torch::Tensor>              tensors;
    std::queue<std::vector<torch::Tensor>> clear_tensors;

    void hold_host(const torch::Tensor& tensor) {
        if (tensor.defined() && tensor.device().is_cpu()) {
            tensors.push_back(tensor);
        }
    }

    void hold(const torch::Tensor& tensor) {
        if (tensor.defined()) {
            tensors.push_back(tensor);
        }
    }

    void release() {
        // Move the current hold set into clear_tensors. Keep two released
        // rounds alive so tensors created for async H2D/D2H copies or CUDA
        // kernels are not freed until the third release point.
        clear_tensors.push(std::move(tensors));
        tensors.clear();
        while (clear_tensors.size() > kReleasedHoldRounds) {
            clear_tensors.pop();
        }
    }

    // Drops every reference owned by this holder, including the rounds kept by
    // release() for asynchronous CUDA work. The caller must first drain that
    // work and synchronize the device.
    TensorHolderClearStats clear() {
        TensorHolderClearStats stats;
        collectStats(tensors, stats);
        tensors.clear();
        while (!clear_tensors.empty()) {
            collectStats(clear_tensors.front(), stats);
            clear_tensors.pop();
        }
        return stats;
    }

private:
    static void collectStats(const std::vector<torch::Tensor>& held_tensors, TensorHolderClearStats& stats) {
        for (const auto& tensor : held_tensors) {
            if (!tensor.defined()) {
                continue;
            }
            ++stats.tensor_count;
            if (tensor.device().is_cpu() && tensor.is_pinned()) {
                ++stats.pinned_tensor_count;
                stats.pinned_bytes += static_cast<size_t>(tensor.numel()) * tensor.element_size();
            }
        }
    }
};

}  // namespace rtp_llm
