# envs/entities.py
import numpy as np
from dataclasses import dataclass, field
from configs.config import cfg


@dataclass
class Entity:
    id: int
    pos: np.ndarray


@dataclass
class UAV(Entity):
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))

    # 类型: 0:Decoy, 1:Strike, 2:Assess (由 config 定义)
    uav_type: int = 0

    # 虽然是 One-Shot，但保留 assigned_target_id 用于可视化或调试
    assigned_target_id: int = -1

    def reset(self, pos, v, u_type):
        self.pos = pos
        self.velocity = v
        self.uav_type = u_type
        self.assigned_target_id = -1


@dataclass
class Target(Entity):
    # 【新想定核心属性】
    # 目标价值 (0.0 ~ 1.0)，越高越值得打
    value: float = 1.0

    # 防御等级 (1.0 ~ 10.0)，越高越难打 (需要更多 Decoy)
    defense_level: float = 1.0

    # 临时记录：这一步有多少 UAV 分配给了我 (用于 env.step 计算奖励)
    # 不再作为 State 输入给网络，而是作为 Step 结算的中间变量
    incoming_uavs: dict = field(default_factory=dict)

    def reset(self, value=None, defense=None):
        """
        重置目标状态。
        如果不指定 value/defense，则随机生成。
        """
        # 价值偏向高分布，避免出现太多 0.1 这种无意义目标
        self.value = value if value is not None else np.random.uniform(0.5, 1.0)

        # 防御等级默认 1~5 (训练初期不宜过高，否则 Phase 1 很难学)
        # 后续可通过 Curriculum Learning 动态调整传入的 defense 参数
        self.defense_level = defense if defense is not None else float(np.random.randint(1, 6))

        self.incoming_uavs = {}