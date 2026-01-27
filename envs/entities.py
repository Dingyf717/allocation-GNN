# envs/entities.py
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Entity:
    id: int
    pos: np.ndarray


@dataclass
class UAV(Entity):
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    # 【新增】类型ID (泛化核心)
    uav_type: int = 0
    # 状态
    assigned_target_id: int = -1
    available: bool = True

    def reset(self, pos, v, u_type):
        self.pos = pos
        self.velocity = v
        self.uav_type = u_type
        self.assigned_target_id = -1
        self.available = True


@dataclass
class Target(Entity):
    # 【新增】需求字典 {type_id: count}
    # 例如 {0: 4, 1: 3, 2: 6}
    demands: Dict[int, int] = field(default_factory=dict)
    # 【新增】已分配计数字典 {type_id: count}
    assigned_counts: Dict[int, int] = field(default_factory=dict)

    # 记录被哪些UAV锁定了 (用于可视化)
    locked_by_uavs: List[int] = field(default_factory=list)

    def reset(self, demand_dict):
        self.demands = demand_dict.copy()
        # 初始化已分配为0
        self.assigned_counts = {k: 0 for k in demand_dict.keys()}
        self.locked_by_uavs = []

    def get_demand_status(self, u_type):
        """
        核心泛化接口：查询当前 UAV 类型对应的需求状态
        返回: (需要的数量, 已分配的数量)
        """
        return self.demands.get(u_type, 0), self.assigned_counts.get(u_type, 0)