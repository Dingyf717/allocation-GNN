# configs/config.py
import numpy as np


class Config:
    # ================= 1. 训练与网络参数 =================
    # 【必须修改】状态向量长度：距离(1)+角度(1)+相对需求(1)+相对已分配(1)+是否满(1)+可用(1)+预留(1) = 7
    STATE_DIM = 7
    SEQ_LEN = 5
    ACTION_DIM = 2  # 0: Skip, 1: Assign

    # Transformer 网络参数
    EMBED_DIM = 128
    NUM_HEADS = 8
    NUM_LAYERS = 2  # 建议保持 2 层

    # PPO 优化参数
    LR_ACTOR = 2e-4
    LR_CRITIC = 1e-3
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    K_EPOCHS = 4
    EPS_CLIP = 0.2
    BATCH_SIZE = 256  # 泛化任务建议调大 Batch Size
    GRAD_NORM_CLIP = 1.0

    # ================= 2. 训练循环控制 (之前遗漏的部分) =================
    # 建议先设为 200 或 500 跑通流程，确认无误后再改为 2000+
    MAX_EPISODES = 2000

    # 每多少轮保存一次 Checkpoint (在 main_train.py 中也有硬编码逻辑，这里可作为参考)
    RESET_EPISODES = 200

    SEED = 42

    # ================= 3. 场景生成配置 (新) =================
    MAP_WIDTH = 1000.0  # 论文场景二地图大小
    MAP_HEIGHT = 1000.0

    # 定义生成函数，支持动态参数
    @staticmethod
    def generate_scenario(num_uavs=None, num_targets=None, num_types=None):
        """
        随机生成一个场景配置字典。
        如果不传参，则使用默认的训练范围随机生成。
        """
        # 训练时随机范围 (增强泛化性)
        # 这里的范围决定了模型训练时的"题目难度"
        if num_uavs is None: num_uavs = np.random.randint(50, 80)
        if num_targets is None: num_targets = np.random.randint(3, 6)
        if num_types is None: num_types = np.random.randint(3, 5)  # 比如 3-5 种类型

        return {
            "n_uavs": num_uavs,
            "n_targets": num_targets,
            "n_types": num_types,
            # 生成类型ID列表 [0, 1, 2, ...]
            "type_ids": list(range(num_types))
        }


cfg = Config()