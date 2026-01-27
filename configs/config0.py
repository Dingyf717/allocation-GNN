# configs/config.py
import numpy as np


class Config:
    # ... (保留前两个阶段的所有参数：物理参数、天气参数) ...
    # 请务必保留 Phase 1 和 Phase 2 添加的 CONSTANTS (PARAM_*, WEATHER_*)

    # ================= 物理模型参数 (回顾) =================
    PARAM_ZETA_D = 150.0
    PARAM_K = 5.0
    PARAM_C1 = 0.75
    PARAM_C2 = 0.25
    PARAM_C3 = 0.75
    PARAM_C4 = 0.25

    # ================= 天气影响参数 (回顾) =================
    WEATHER_LEVEL_RANGE = (1, 6)
    WEATHER_FACTOR_TYPE1 = np.array([
        [0.85, 0.85, 0.85, 0.85, 0.75],
        [0.85, 0.85, 0.85, 0.85, 0.75],
        [0.85, 0.85, 0.85, 0.85, 0.75],
        [0.85, 0.85, 0.85, 0.85, 0.75],
        [0.75, 0.75, 0.75, 0.75, 0.75]
    ])
    WEATHER_FACTOR_TYPE2 = np.array([
        [0.9, 0.9, 0.9, 0.8, 0.8],
        [0.9, 0.9, 0.9, 0.8, 0.8],
        [0.9, 0.9, 0.9, 0.8, 0.8],
        [0.9, 0.9, 0.9, 0.8, 0.8],
        [0.8, 0.8, 0.8, 0.8, 0.8]
    ])

    # ================= 仿真环境参数 =================
    MAP_WIDTH = 180.0
    MAP_HEIGHT = 160.0
    UAV_GEN_X_RANGE = (0, 30)
    TARGET_GEN_X_RANGE = (160, 180)
    NUM_UAVS = 30
    NUM_TARGETS = 10
    NUM_NFZ = 2
    NUM_INTERCEPTORS = 2
    INTERCEPT_RAD = 3.0

    # 成本与权重
    COST_WEIGHT_OMEGA = 0.0

    # 这些常数用于 Phase 2 的环境生成 (Partial Mod)
    WEATHER_SPEED_FACTOR = 0.85
    WEATHER_LOAD_FACTOR = 0.90

    # ================= PPO & 网络参数 (Table I) =================
    STATE_DIM = 14
    SEQ_LEN = 5  # 5-step temporal window
    ACTION_DIM = 2  # Assign or Skip

    # Transformer 参数
    EMBED_DIM = 128  # Channels of attention head
    NUM_HEADS = 8  # Attention heads
    NUM_LAYERS = 2  # 建议值，保持适中深度

    # 训练超参数
    LR_ACTOR = 2e-4  # Table I
    LR_CRITIC = 1e-3  # Table I
    GAMMA = 0.998  # Table I
    GAE_LAMBDA = 0.95  # Table I

    K_EPOCHS = 5  # PPO 更新次数
    EPS_CLIP = 0.2  # PPO Clip
    BATCH_SIZE = 64  # Table I

    GRAD_NORM_CLIP = 1.0  # Table I: Gradient threshold

    MAX_EPISODES = 2000  # Table I
    RESET_EPISODES = 200  # Table I: Episodes of environment resets

    SEED = 42


cfg = Config()