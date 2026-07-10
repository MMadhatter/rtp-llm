from rtp_llm.server.server_args.util import str2bool


def init_engine_group_args(parser, runtime_config):
    ##############################################################################################################
    # Engine Configuration
    # Fields merged from EngineConfig to RuntimeConfig (warm_up, warm_up_with_loss)
    ##############################################################################################################
    engine_group = parser.add_argument_group("Engine Configuration")
    engine_group.add_argument(
        "--warm_up",
        env_name="WARM_UP",
        bind_to=(runtime_config, "warm_up"),
        type=str2bool,
        default=True,
        help="在服务启动时是否开启预热",
    )
    engine_group.add_argument(
        "--warm_up_with_loss",
        env_name="WARM_UP_WITH_LOSS",
        bind_to=(runtime_config, "warm_up_with_loss"),
        type=str2bool,
        default=False,
        help="在服务启动时是否开启损失去预热",
    )
    engine_group.add_argument(
        "--enable_sleep_mode",
        "--enable-sleep-mode",
        env_name="ENABLE_SLEEP_MODE",
        bind_to=(
            (runtime_config, "enable_sleep_mode")
            if hasattr(runtime_config, "enable_sleep_mode")
            else None
        ),
        type=str2bool,
        default=False,
        help="是否开启 sleep/wake_up 生命周期管理接口，默认关闭",
    )
    engine_group.add_argument(
        "--sleep_mode_level",
        "--sleep-mode-level",
        env_name="SLEEP_MODE_LEVEL",
        bind_to=(
            (runtime_config, "sleep_mode_level")
            if hasattr(runtime_config, "sleep_mode_level")
            else None
        ),
        type=int,
        default=1,
        help="本进程启动时选定的 sleep level（torch_memory_saver 在加载时就绑定权重 region 的 "
        "cpu_backup，无法按请求切换）。1=sleep 时权重备份到 pinned host（唤醒快，常驻 host 内存）；"
        "2=sleep 时丢弃权重（释放 GPU+host），唤醒时从本地磁盘 raw 备份恢复。/sleep 请求的 level "
        "必须与此值一致，默认 1",
    )
