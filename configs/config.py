# configs/config.py
import numpy as np


class Config:
    # ================= 1. 训练与网络参数 =================
    # 【改动】移除序列长度，改为定义节点特征维度
    # SEQ_LEN = 5  <-- 删除或注释掉

    # UAV 特征: [x, y, vx, vy, type, available] (归一化后)
    UAV_STATE_DIM = 6

    # Target 特征: [x, y, demand_type_0, demand_type_1, demand_type_2]
    # 假设最大支持 3 种类型 (0, 1, 2)，如果更多可以调大
    MAX_TYPES = 3
    TARGET_STATE_DIM = 2 + MAX_TYPES

    # Action 保持不变，或者在后续改为输出矩阵时再调整
    ACTION_DIM = 2

    # Transformer/GNN 网络参数
    EMBED_DIM = 128
    NUM_HEADS = 8
    NUM_LAYERS = 2  # Encoder 层数

    # PPO 优化参数 (保持不变)
    LR_ACTOR = 2e-4
    LR_CRITIC = 1e-3
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    K_EPOCHS = 4
    EPS_CLIP = 0.2
    BATCH_SIZE = 64  # Deep Scheme 因为是一次性处理所有，BatchSize 可以适当减小
    GRAD_NORM_CLIP = 1.0

    # ================= 2. 训练循环控制 =================
    MAX_EPISODES = 2000
    RESET_EPISODES = 200
    SEED = 42

    # ================= 3. 场景生成配置 =================
    MAP_WIDTH = 1000.0
    MAP_HEIGHT = 1000.0

    @staticmethod
    def generate_scenario(num_uavs=None, num_targets=None, num_types=None):
        if num_uavs is None: num_uavs = np.random.randint(30, 50)
        if num_targets is None: num_targets = np.random.randint(3, 4)
        if num_types is None: num_types = np.random.randint(2, 4)  # 确保不超过 MAX_TYPES

        return {
            "n_uavs": num_uavs,
            "n_targets": num_targets,
            "n_types": num_types,
            "type_ids": list(range(num_types))
        }


cfg = Config()