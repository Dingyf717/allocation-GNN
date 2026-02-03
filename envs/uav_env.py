# envs/uav_env.py
import gym
import numpy as np
from gym import spaces
from configs.config import cfg
from envs.entities import UAV, Target


class UAVEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(cfg.ACTION_DIM)

        # 【Stage 3 新增】扩展观察空间
        # 现在的 Obs 是一个包含节点和边信息的图
        self.observation_space = spaces.Dict({
            # UAV 节点特征 (N, 7)
            "uavs": spaces.Box(low=-np.inf, high=np.inf, shape=(cfg.UAV_STATE_DIM,), dtype=np.float32),
            # Target 节点特征 (M, 4)
            "targets": spaces.Box(low=-np.inf, high=np.inf, shape=(cfg.TARGET_STATE_DIM,), dtype=np.float32),
            # 边特征 (N, M, 2) -> (Distance, Heading_Cos)
            "edges": spaces.Box(low=-np.inf, high=np.inf, shape=(cfg.EDGE_DIM,), dtype=np.float32)
        })

        self.uavs = []
        self.targets = []

    def reset(self, full_reset=True):
        if full_reset:
            scen = cfg.generate_scenario()
            self._init_scene(scen)
        return self._get_obs()

    def _init_scene(self, scen):
        self.uavs = []
        self.targets = []

        # 初始化 UAVs (随机兵种)
        for i in range(scen['n_uavs']):
            pos = np.random.rand(2) * cfg.MAP_WIDTH
            angle = np.random.uniform(0, 2 * np.pi)
            vel = np.array([np.cos(angle), np.sin(angle)]) * 15.0
            u_type = np.random.choice([cfg.TYPE_DECOY, cfg.TYPE_STRIKE, cfg.TYPE_ASSESS])

            uav = UAV(id=i, pos=pos)
            uav.reset(pos, vel, u_type)
            self.uavs.append(uav)

        # 初始化 Targets
        for i in range(scen['n_targets']):
            pos = np.random.rand(2) * cfg.MAP_WIDTH
            tgt = Target(id=i, pos=pos)
            tgt.reset()
            self.targets.append(tgt)

    def _get_obs(self):
        """
        【Stage 3 核心】构建图数据
        返回: Dict {'uavs': (N,7), 'targets': (M,4), 'edges': (N,M,2)}
        """
        # 1. UAV Matrix (N, 7)
        uav_mat = []
        for u in self.uavs:
            norm_x = u.pos[0] / cfg.MAP_WIDTH
            norm_y = u.pos[1] / cfg.MAP_HEIGHT
            norm_vx = u.velocity[0] / 20.0
            norm_vy = u.velocity[1] / 20.0
            type_vec = [0, 0, 0]
            type_vec[u.uav_type] = 1
            uav_mat.append([norm_x, norm_y, norm_vx, norm_vy] + type_vec)

        # 2. Target Matrix (M, 4)
        tgt_mat = []
        for t in self.targets:
            norm_x = t.pos[0] / cfg.MAP_WIDTH
            norm_y = t.pos[1] / cfg.MAP_HEIGHT
            norm_def = t.defense_level / 10.0
            tgt_mat.append([norm_x, norm_y, t.value, norm_def])

        # 3. Edge Tensor (N, M, 2)
        # 维度 0: Normalized Distance
        # 维度 1: Heading Cosine (速度方向与连线的夹角余弦)
        n_u = len(self.uavs)
        n_t = len(self.targets)
        edge_mat = np.zeros((n_u, n_t, cfg.EDGE_DIM), dtype=np.float32)

        # 归一化系数 (地图对角线)
        max_dist = cfg.MAP_WIDTH * 1.414

        for i in range(n_u):
            u = self.uavs[i]
            for j in range(n_t):
                t = self.targets[j]

                # 相对位置向量
                rel_pos = t.pos - u.pos
                dist = np.linalg.norm(rel_pos)

                # --- Feature 1: Distance ---
                edge_mat[i, j, 0] = dist / max_dist

                # --- Feature 2: Heading Alignment ---
                # 计算 UAV速度向量 与 目标方向向量 的余弦相似度
                # 1.0 = 正对目标飞, -1.0 = 背对目标飞
                if dist > 1e-5 and np.linalg.norm(u.velocity) > 1e-5:
                    # dot(a, b) / (|a| * |b|)
                    cos_sim = np.dot(u.velocity, rel_pos) / (np.linalg.norm(u.velocity) * dist)
                else:
                    cos_sim = 0.0

                edge_mat[i, j, 1] = cos_sim

        return {
            "uavs": np.array(uav_mat, dtype=np.float32),
            "targets": np.array(tgt_mat, dtype=np.float32),
            "edges": edge_mat
        }

    def step(self, action):
        """
        Stage 2 + 3: 逻辑保持不变，但返回值适配新的 obs 结构
        """
        # --- 1. 统计分配 (Aggregation) ---
        target_allocations = {j: {cfg.TYPE_DECOY: 0, cfg.TYPE_STRIKE: 0, cfg.TYPE_ASSESS: 0}
                              for j in range(len(self.targets))}

        total_dist = 0.0
        active_uav_count = 0

        for u_idx, t_idx in enumerate(action):
            uav = self.uavs[u_idx]
            uav.assigned_target_id = t_idx

            if t_idx == 0: continue  # Loiter

            real_t_idx = t_idx - 1
            if real_t_idx < len(self.targets):
                target_allocations[real_t_idx][uav.uav_type] += 1
                dist = np.linalg.norm(uav.pos - self.targets[real_t_idx].pos)
                total_dist += dist
                active_uav_count += 1

        # --- 2. 计算奖励 (Gate Logic) ---
        mission_reward = 0.0
        rho = 0.6
        step_info_details = []

        for j, counts in target_allocations.items():
            tgt = self.targets[j]
            n_d, n_s, n_a = counts[cfg.TYPE_DECOY], counts[cfg.TYPE_STRIKE], counts[cfg.TYPE_ASSESS]

            d_val = max(1.0, tgt.defense_level)
            p_pen = min(1.0, n_d / d_val)  # 突防
            p_dmg = 1.0 - (1.0 - rho) ** n_s  # 毁伤
            i_info = 1.0 if n_a >= 1 else 0.0  # 闭环

            task_r = tgt.value * p_pen * p_dmg * i_info * 10.0
            mission_reward += task_r

            step_info_details.append({"tgt_id": j, "alloc": counts, "reward": task_r})

        # --- 3. 移动惩罚 ---
        dist_penalty = 0.0
        if active_uav_count > 0:
            avg_dist = total_dist / active_uav_count
            dist_penalty = (avg_dist / (cfg.MAP_WIDTH * 1.414)) * 1.0

        total_reward = mission_reward - dist_penalty
        done = True

        info = {
            "mission_reward": mission_reward,
            "dist_penalty": dist_penalty,
            "details": step_info_details,
            "sat_rate": 0.0  # 占位
        }

        return self._get_obs(), total_reward, done, info