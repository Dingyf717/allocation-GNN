# configs/config.py
import numpy as np


class Config:
    # ================= 1. 实体类型定义 =================
    TYPE_DECOY = 0  # 诱饵: 消耗防御
    TYPE_STRIKE = 1  # 打击: 输出火力
    TYPE_ASSESS = 2  # 评估: 闭环信息
    NUM_UAV_TYPES = 3

    # ================= 2. 状态维度定义 (GNN Input) =================
    # UAV State: [x, y, vx, vy, is_decoy, is_strike, is_assess]
    # Pos(2) + Vel(2) + OneHot(3) = 7
    # 移除了油量/available等非核心特征，专注拓扑决策
    UAV_STATE_DIM = 7

    # Target State: [x, y, value, defense_level]
    # Pos(2) + Value(1) + Defense(1) = 4
    TARGET_STATE_DIM = 4

    # Edge Feature: [distance, angle]
    # 这是 GNN 能够感知"代价"的关键
    EDGE_DIM = 2
    ACTION_DIM = 20

    # ================= 3. 网络与训练参数 =================
    EMBED_DIM = 128
    NUM_HEADS = 8
    NUM_LAYERS = 2

    # PPO 参数
    LR_ACTOR = 2e-4
    LR_CRITIC = 1e-3
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    K_EPOCHS = 4
    EPS_CLIP = 0.2
    BATCH_SIZE = 128  # 稍微调大 Batch 以稳定梯度
    GRAD_NORM_CLIP = 1.0

    # 训练循环
    MAX_EPISODES = 100000

    # ================= 4. 场景生成 =================
    MAP_WIDTH = 1000.0
    MAP_HEIGHT = 1000.0

    @staticmethod
    def generate_scenario(num_uavs=None, num_targets=None):
        """
        随机生成场景配置
        不再需要生成 'type_ids' 列表，因为现在兵种是固定的三种功能
        """
        if num_uavs is None: num_uavs = np.random.randint(20, 40)
        if num_targets is None: num_targets = np.random.randint(3, 8)

        return {
            "n_uavs": num_uavs,
            "n_targets": num_targets
        }


cfg = Config()