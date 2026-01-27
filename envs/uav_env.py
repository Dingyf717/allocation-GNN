# envs/uav_env.py
import gym
import numpy as np
from gym import spaces
from collections import deque
from configs.config import cfg
from envs.entities import UAV, Target
from envs.mechanics import calc_reward, get_distance, calc_angle_score


class UAVEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(cfg.ACTION_DIM)
        # 7维状态向量
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(cfg.SEQ_LEN, cfg.STATE_DIM), dtype=np.float32
        )
        self.state_buffer = deque(maxlen=cfg.SEQ_LEN)

    def reset(self, full_reset=True):
        if full_reset:
            # 1. 生成随机场景参数 (泛化！)
            # 也可以在这里指定论文的具体参数 (80 uavs, 4 targets, 4 types)
            scen = cfg.generate_scenario()
            self._init_scene(scen)
        else:
            self._reset_state_only()

        self.uav_idx = 0
        self.target_idx = 0
        self.state_buffer.clear()
        for _ in range(cfg.SEQ_LEN):
            self.state_buffer.append(np.zeros(cfg.STATE_DIM))
        return self._get_obs()

    def _init_scene(self, scen):
        self.uavs = []
        self.targets = []

        # 初始化 UAVs
        for i in range(scen['n_uavs']):
            pos = np.random.rand(2) * 250.0  # 出发区
            # 随机速度方向
            angle = np.random.uniform(0, 2 * np.pi)
            vel = np.array([np.cos(angle), np.sin(angle)]) * 15.0
            # 随机分配类型
            u_type = np.random.choice(scen['type_ids'])

            uav = UAV(id=i, pos=pos)
            uav.reset(pos, vel, u_type)
            self.uavs.append(uav)

        # 初始化 Targets
        for i in range(scen['n_targets']):
            pos = np.random.rand(2) * 1000.0
            # 随机生成需求
            # 确保总需求略小于或等于总UAV数，或者允许溢出
            demands = {}
            for t_id in scen['type_ids']:
                demands[t_id] = np.random.randint(3, 4)  # 随机需求量

            tgt = Target(id=i, pos=pos)
            tgt.reset(demands)
            self.targets.append(tgt)

    def _get_obs(self):
        # 如果遍历结束
        if self.uav_idx >= len(self.uavs):
            return np.zeros((cfg.SEQ_LEN, cfg.STATE_DIM))

        uav = self.uavs[self.uav_idx]
        target = self.targets[self.target_idx]

        # --- 构建相对特征状态 (7维) ---
        # 1. 物理特征
        dist = get_distance(uav.pos, target.pos) / 1000.0  # 归一化
        angle_score = calc_angle_score(uav.velocity, uav.pos, target.pos)

        # 2. 供需特征 (泛化核心)
        # "目标需要多少个*我这种类型*的人？"
        needed, assigned = target.get_demand_status(uav.uav_type)

        # 归一化处理 (假设最大需求一般不超过20，或者用log缩放)
        feat_needed = needed / 10.0
        feat_assigned = assigned / 10.0
        feat_is_satisfied = 1.0 if needed <= 0 else 0.0

        # 3. 自身状态
        feat_uav_available = 1.0 if uav.available else 0.0

        # 组合
        state = np.array([
            dist,  # 距离
            angle_score,  # 角度优势
            feat_needed,  # 缺口 (Relative Demand)
            feat_assigned,  # 已填 (Relative Assigned)
            feat_is_satisfied,  # 坑满了吗
            feat_uav_available,  # 我还能动吗
            0.0  # 预留位 (可以放全局进度等)
        ], dtype=np.float32)

        self.state_buffer.append(state)
        return np.array(self.state_buffer)

    def step(self, action):
        uav = self.uavs[self.uav_idx]
        target = self.targets[self.target_idx]

        # 计算奖励
        reward = calc_reward(uav, target, action)

        info = {}  # 初始化 info

        # 执行状态更新
        if action == 1:
            needed, _ = target.get_demand_status(uav.uav_type)
            if needed > 0:
                # 只有真正有需求且匹配时，才更新状态
                target.demands[uav.uav_type] -= 1
                target.assigned_counts[uav.uav_type] += 1
                uav.assigned_target_id = target.id
                uav.available = False

                # 【新增】标记这是一个有效分配
                info['is_valid_action'] = True

                # 【修复 2】计算并返回角度得分，让 main_train.py 能统计到 Ang
                # 需要调用 mechanics 中的函数
                current_angle_score = calc_angle_score(uav.velocity, uav.pos, target.pos)
                info['angle_score'] = current_angle_score

                # 成功分配后，跳到下一个 UAV
                self.uav_idx += 1
                self.target_idx = 0
            else:
                # 惩罚性操作：虽然选了 Assign 但无效
                # 逻辑流转：视为被迫 Skip，看下一个 Target
                info['is_valid_action'] = False
                info['angle_score'] = 0.0  # 无效分配没有角度分
                self.target_idx += 1
        else:
            # Skip
            info['is_valid_action'] = False
            self.target_idx += 1

        # 边界检查
        if self.target_idx >= len(self.targets):
            self.uav_idx += 1
            self.target_idx = 0

        done = (self.uav_idx >= len(self.uavs))

        # 获取下一个状态
        obs = self._get_obs()
        return obs, reward, done, info