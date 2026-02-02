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
        # 动作空间: 0 ~ N_targets (其中 N_targets 代表 Skip)
        # 注意: 这里只定义维度，具体 range 在 step 中处理
        self.action_space = spaces.Discrete(cfg.ACTION_DIM)

        # 观察空间: 变长 Dict
        self.observation_space = spaces.Dict({
            "uavs": spaces.Box(low=-np.inf, high=np.inf, shape=(cfg.UAV_STATE_DIM,), dtype=np.float32),
            "targets": spaces.Box(low=-np.inf, high=np.inf, shape=(cfg.TARGET_STATE_DIM,), dtype=np.float32)
        })

    def reset(self, full_reset=True):
        if full_reset:
            scen = cfg.generate_scenario()
            self._init_scene(scen)
        else:
            self._reset_state_only()
        return self._get_obs()

    def _init_scene(self, scen):
        self.uavs = []
        self.targets = []
        # 初始化 UAVs
        for i in range(scen['n_uavs']):
            pos = np.random.rand(2) * cfg.MAP_WIDTH
            angle = np.random.uniform(0, 2 * np.pi)
            vel = np.array([np.cos(angle), np.sin(angle)]) * 15.0
            u_type = np.random.choice(scen['type_ids'])
            uav = UAV(id=i, pos=pos)
            uav.reset(pos, vel, u_type)
            self.uavs.append(uav)
        # 初始化 Targets
        for i in range(scen['n_targets']):
            pos = np.random.rand(2) * cfg.MAP_WIDTH
            demands = {}
            for t_id in scen['type_ids']:
                demands[t_id] = np.random.randint(3, 4)
            tgt = Target(id=i, pos=pos)
            tgt.reset(demands)
            self.targets.append(tgt)

    def _reset_state_only(self):
        pass

    def _get_obs(self):
        # 1. UAV Matrix
        uav_list = []
        for u in self.uavs:
            norm_x = u.pos[0] / cfg.MAP_WIDTH
            norm_y = u.pos[1] / cfg.MAP_HEIGHT
            norm_vx = u.velocity[0] / 15.0
            norm_vy = u.velocity[1] / 15.0
            u_type = float(u.uav_type)
            avail = 1.0 if u.available else 0.0
            uav_list.append([norm_x, norm_y, norm_vx, norm_vy, u_type, avail])

        # 2. Target Matrix
        target_list = []
        for t in self.targets:
            norm_x = t.pos[0] / cfg.MAP_WIDTH
            norm_y = t.pos[1] / cfg.MAP_HEIGHT
            demands_vec = [0.0] * cfg.MAX_TYPES
            for type_id, count in t.demands.items():
                if type_id < cfg.MAX_TYPES:
                    demands_vec[type_id] = float(count) / 10.0
            target_list.append([norm_x, norm_y] + demands_vec)

        return np.array(uav_list, dtype=np.float32), np.array(target_list, dtype=np.float32)

    def step(self, action):
        """
        Step 5 重写: Global One-Shot Step
        参数:
            action: (N_uav,) 的整数数组，表示每个 UAV 选择了哪个 Target ID
                    如果 action[i] == N_targets，表示 Skip
        """
        total_reward = 0
        valid_assigns = 0
        assigned_pairs = []

        n_targets = len(self.targets)

        # 遍历每个 UAV 的决策
        for u_idx, t_idx in enumerate(action):
            uav = self.uavs[u_idx]

            # 1. 判断是否 Skip
            if t_idx == n_targets:
                # Skip 只有微小惩罚或无惩罚
                step_r = -0.1
            else:
                # 2. 尝试分配给 Target t_idx
                # 越界保护 (虽然网络不应该输出越界)
                if t_idx >= n_targets:
                    step_r = -1.0  # Error penalty
                else:
                    target = self.targets[t_idx]

                    # 检查距离 (为了计算分数)
                    dist = get_distance(uav.pos, target.pos)
                    dist_penalty = -0.001 * dist
                    angle_score = calc_angle_score(uav.velocity, uav.pos, target.pos)
                    angle_reward = 0.5 * angle_score

                    # 检查供需
                    needed, _ = target.get_demand_status(uav.uav_type)

                    if needed > 0:
                        # === 成功分配 ===
                        step_r = 10.0 + angle_reward + dist_penalty
                        # 更新状态 (扣减需求)
                        target.demands[uav.uav_type] -= 1
                        target.assigned_counts[uav.uav_type] += 1
                        uav.available = False

                        valid_assigns += 1
                        assigned_pairs.append((u_idx, t_idx))
                    else:
                        # === 无效分配 (不需要或已满) ===
                        step_r = -5.0  # 惩罚乱选

            total_reward += step_r

        # 计算满足率
        total_needed = 0
        total_filled = 0
        for t in self.targets:
            total_needed += (sum(t.demands.values()) + sum(t.assigned_counts.values()))
            total_filled += sum(t.assigned_counts.values())

        sat_rate = total_filled / total_needed if total_needed > 0 else 0.0

        info = {
            "valid_assigns": valid_assigns,
            "sat_rate": sat_rate,
            "assigned_pairs": assigned_pairs
        }

        # One-Shot 任务，一步即结束
        done = True

        # 返回 next_obs (对于 One-Shot 其实没用，但为了兼容性返回当前状态)
        next_obs = self._get_obs()

        return next_obs, total_reward, done, info