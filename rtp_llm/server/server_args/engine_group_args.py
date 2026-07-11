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
    engine_group.add_argument(
        "--sleep_l2_backup_dir",
        "--sleep-l2-backup-dir",
        env_name="SLEEP_L2_BACKUP_DIR",
        bind_to=(
            (runtime_config, "sleep_l2_backup_dir")
            if hasattr(runtime_config, "sleep_l2_backup_dir")
            else None
        ),
        type=str,
        default="",
        help="level-2 sleep（丢弃权重）时 raw 权重备份的本地磁盘根目录，应指向本地 NVMe/SSD；"
        "备份大小约等于权重大小（大模型可达 ~1TB）。实际会在其下按 <pid> 建子目录以隔离同机共存的实例"
        "（如同机 prefill+decode，rank tag 可能相同会冲突）。备份用完即弃：wake 恢复成功后自动删除该 "
        "<pid> 子目录，不留常驻磁盘副本，下次 sleep 重新 dump。不设置时回退到 /tmp/rtp_llm_sleep_l2 并告警",
    )
