# test_compare_hungarian.py
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment  # 核心：匈牙利算法库

# 解决 OpenMP 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from configs.config import cfg
from envs.uav_env import UAVEnv
from agents.ppo import PPOAgent
from envs.mechanics import calc_angle_score, get_distance

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def run_ppo_inference(env, agent):
    """运行现有的 PPO 算法进行推理"""
    start_time = time.time()

    # 必须确保环境处于初始状态，但不要再次 reset (因为要保证场景一致)
    # 我们假设传入的 env 已经是 reset 好的

    state = env._get_obs()  # 获取初始状态
    done = False
    matches = []
    total_reward = 0.0

    # 临时变量用于统计
    satisfied_counts = {t.id: 0 for t in env.targets}

    while not done:
        # 获取当前决策对象
        if env.uav_idx >= len(env.uavs): break
        uav = env.uavs[env.uav_idx]
        tgt = env.targets[env.target_idx]

        # 1. 模型预测
        if hasattr(agent, 'predict'):
            action = agent.predict(state)
        else:
            action = agent.select_action(state)

        # 2. 环境交互
        next_state, reward, done, info = env.step(action)

        # 3. 记录有效分配
        if action == 1 and info.get('is_valid_action', False):
            matches.append((uav.id, tgt.id))
            satisfied_counts[tgt.id] += 1

        total_reward += reward
        state = next_state

    duration = time.time() - start_time

    # 计算满足率
    total_demand = sum([sum(t.demands.values()) for t in env.targets])
    total_filled = sum([sum(t.assigned_counts.values()) for t in env.targets])
    sat_rate = total_filled / total_demand if total_demand > 0 else 0

    return {
        "matches": matches,
        "reward": total_reward,
        "sat_rate": sat_rate,
        "time": duration,
        "name": "PPO (RL)"
    }


def run_hungarian_inference(env):
    """
    运行匈牙利算法 (KM算法)
    原理：构建 (N_uav, N_demand_slots) 的收益矩阵，求解最大权匹配
    """
    start_time = time.time()

    uavs = env.uavs
    targets = env.targets

    # --- 1. 构建需求槽位 (Column Expansion) ---
    # 匈牙利算法是 1对1 匹配。如果 Target 0 需要 2个 Type 0，我们需要生成 2列。
    demand_slots = []  # 存储 (Target_Index, Required_Type)

    for t_idx, tgt in enumerate(targets):
        for type_id, count in tgt.demands.items():
            for _ in range(count):
                demand_slots.append({
                    "t_idx": t_idx,
                    "req_type": type_id,
                    "target_obj": tgt
                })

    n_uavs = len(uavs)
    n_slots = len(demand_slots)

    if n_slots == 0:
        return {"matches": [], "reward": 0, "sat_rate": 0, "time": 0, "name": "Hungarian"}

    # --- 2. 构建收益矩阵 (Profit Matrix) ---
    # 行：UAV, 列：需求槽位
    # Scipy 的 linear_sum_assignment 是求最小成本，所以我们填入 "负收益"
    cost_matrix = np.zeros((n_uavs, n_slots))

    # 预设一个巨大的惩罚值，用于处理类型不匹配
    INF_COST = 1e6

    for i, uav in enumerate(uavs):
        for j, slot in enumerate(demand_slots):
            tgt = slot['target_obj']

            # [约束 1] 类型匹配检查
            if uav.uav_type != slot['req_type']:
                cost_matrix[i, j] = INF_COST  # 类型不符，禁止匹配
                continue

            # [计算收益] 复用 mechanics.py 中的逻辑
            # Reward = 10.0 + Angle_Score*0.5 - Dist*0.001
            dist = get_distance(uav.pos, tgt.pos)
            angle_score = calc_angle_score(uav.velocity, uav.pos, tgt.pos)

            # 计算正向收益
            # 注意：这里我们只计算匹配成功的收益，忽略 Skip 的微小惩罚，因为匈牙利算法只看匹配
            score = 10.0 + 0.5 * angle_score - 0.001 * dist

            # 转为成本 (取负)
            cost_matrix[i, j] = -score

    # --- 3. 求解 (Solve) ---
    # row_ind 是 UAV 索引, col_ind 是 Slot 索引
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # --- 4. 统计结果 ---
    matches = []
    total_reward = 0.0
    valid_matches_count = 0

    for r, c in zip(row_ind, col_ind):
        # 检查是否是有效匹配 (成本不是 INF)
        if cost_matrix[r, c] < (INF_COST / 2):
            u_id = uavs[r].id
            slot = demand_slots[c]
            t_id = slot['target_obj'].id

            matches.append((u_id, t_id))

            # 还原真实收益
            reward = -cost_matrix[r, c]
            total_reward += reward
            valid_matches_count += 1

    # 注意：匈牙利算法算出的 total_reward 是纯匹配收益
    # 为了和 RL 公平对比，未匹配的 UAV (Skip) 理论上也有 -0.1 的惩罚
    # 但由于数量级差异 (10.0 vs 0.1)，通常可以忽略，或者手动补上
    num_skipped = n_uavs - valid_matches_count
    total_reward += num_skipped * (-0.1)

    duration = time.time() - start_time

    # 计算满足率
    total_demand = len(demand_slots)
    sat_rate = valid_matches_count / total_demand if total_demand > 0 else 0

    return {
        "matches": matches,
        "reward": total_reward,
        "sat_rate": sat_rate,
        "time": duration,
        "name": "Hungarian (Optimal)"
    }


def visualize_comparison(env, res_ppo, res_hung):
    """画对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    def plot_on_ax(ax, res, title):
        # 画地图元素
        ax.set_xlim(0, cfg.MAP_WIDTH)
        ax.set_ylim(0, cfg.MAP_HEIGHT)
        ax.set_title(title)

        # 画目标
        for tgt in env.targets:
            ax.scatter(tgt.pos[0], tgt.pos[1], c='orange', marker='*', s=200, edgecolors='black', zorder=5)
            ax.text(tgt.pos[0], tgt.pos[1] + 15, f"T{tgt.id}", ha='center')

        # 画无人机
        for uav in env.uavs:
            # 检查是否在匹配列表中
            is_matched = any(uav.id == m[0] for m in res['matches'])
            color = 'blue' if is_matched else 'gray'
            alpha = 1.0 if is_matched else 0.3
            ax.scatter(uav.pos[0], uav.pos[1], c=color, marker='^', s=60, alpha=alpha, zorder=3)

        # 画连线
        for (u_id, t_id) in res['matches']:
            u = next(u for u in env.uavs if u.id == u_id)
            t = next(t for t in env.targets if t.id == t_id)
            ax.plot([u.pos[0], t.pos[0]], [u.pos[1], t.pos[1]], 'k--', alpha=0.5, linewidth=1)

        # 添加统计信息文本
        info_text = (f"Reward: {res['reward']:.2f}\n"
                     f"Sat Rate: {res['sat_rate'] * 100:.1f}%\n"
                     f"Time: {res['time'] * 1000:.1f} ms")
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, va='top',
                bbox=dict(boxstyle="round", fc="white", alpha=0.9))

    plot_on_ax(ax1, res_ppo, f"PPO Agent (Seq)\nMatches: {len(res_ppo['matches'])}")
    plot_on_ax(ax2, res_hung, f"Hungarian Alg (Global)\nMatches: {len(res_hung['matches'])}")

    plt.tight_layout()
    plt.savefig("comparison_result.png")
    print("对比结果已保存至 comparison_result.png")
    plt.show()


def main():
    # 1. 准备环境和模型
    model_dir = "./saved_models"
    # 简化的模型加载逻辑，寻找最新的
    if not os.path.exists(model_dir):
        print("错误: 找不到模型目录")
        return

    all_subdirs = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if
                   os.path.isdir(os.path.join(model_dir, d))]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    model_path = os.path.join(latest_subdir, "checkpoint_ep2000.pth")
    if not os.path.exists(model_path): model_path = os.path.join(latest_subdir, "final_model.pth")
    if not os.path.exists(model_path):
        # 尝试找任意一个 best_model
        model_path = os.path.join(latest_subdir, "best_model.pth")

    print(f"Loading Model: {model_path}")

    env = UAVEnv()
    agent = PPOAgent()
    try:
        ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
    except:
        ckpt = torch.load(model_path, map_location='cpu')

    agent.policy.load_state_dict(ckpt)
    agent.policy.eval()

    # 2. 生成一个固定场景 (用于公平对比)
    # 我们通过设置种子或临时覆盖 generate_scenario 来控制
    # 这里为了演示，生成一个中等规模场景
    def custom_scenario():
        return {
            "n_uavs": 500,
            "n_targets": 10,
            "n_types": 3,
            "type_ids": [0, 1, 2]
        }

    # 备份并覆盖
    old_func = cfg.generate_scenario
    cfg.generate_scenario = custom_scenario

    # --- 初始化 PPO 环境 ---
    env.reset(full_reset=True)

    # 深度拷贝场景信息，以便给匈牙利算法重置用
    # 由于 gym env 的 reset 比较复杂，我们采取这样的策略：
    # 先跑 PPO，跑完后，我们手动把 env 里的 uavs 和 targets 的状态恢复到初始值
    # 或者，我们保存初始状态的快照

    import copy
    initial_uavs = copy.deepcopy(env.uavs)
    initial_targets = copy.deepcopy(env.targets)

    print("-" * 60)
    print("开始运行 PPO 算法 (Sequential)...")
    res_ppo = run_ppo_inference(env, agent)
    print(f"PPO 完成: Reward={res_ppo['reward']:.2f}, Time={res_ppo['time']:.4f}s")

    # --- 重置环境给匈牙利算法 ---
    # 恢复状态
    env.uavs = copy.deepcopy(initial_uavs)
    env.targets = copy.deepcopy(initial_targets)
    # 重置索引
    env.uav_idx = 0
    env.target_idx = 0

    print("-" * 60)
    print("开始运行 Hungarian 算法 (Global Optimization)...")
    res_hung = run_hungarian_inference(env)
    print(f"Hungarian 完成: Reward={res_hung['reward']:.2f}, Time={res_hung['time']:.4f}s")
    print("-" * 60)

    # 3. 结果对比与可视化
    print(f"{'Metric':<15} | {'PPO':<15} | {'Hungarian':<15} | {'Diff'}")
    print("-" * 60)
    print(
        f"{'Total Reward':<15} | {res_ppo['reward']:<15.2f} | {res_hung['reward']:<15.2f} | {(res_ppo['reward'] - res_hung['reward']):.2f}")
    print(
        f"{'Sat Rate':<15} | {res_ppo['sat_rate'] * 100:<14.1f}% | {res_hung['sat_rate'] * 100:<14.1f}% | {(res_ppo['sat_rate'] - res_hung['sat_rate']) * 100:.1f}%")
    print(
        f"{'Time (s)':<15} | {res_ppo['time']:<15.4f} | {res_hung['time']:<15.4f} | x{res_ppo['time'] / max(1e-6, res_hung['time']):.1f}")

    visualize_comparison(env, res_ppo, res_hung)

    # 还原配置
    cfg.generate_scenario = old_func


if __name__ == "__main__":
    main()