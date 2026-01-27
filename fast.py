# fast.py (已修复版)
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

# 解决 OpenMP 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from configs.config import cfg
from envs.uav_env import UAVEnv
from agents.ppo import PPOAgent
from envs.mechanics import get_distance

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def batch_inference(env, agent):
    """
    Batch 推理核心函数：
    1. 构造所有 (UAV, Target) 的状态矩阵
    2. 并行推理得到打分
    3. 排序并贪婪分配
    """
    n_uavs = len(env.uavs)
    n_targets = len(env.targets)

    # --- 1. 向量化计算特征 (Vectorized State Generation) ---
    uav_pos = np.array([u.pos for u in env.uavs])
    tgt_pos = np.array([t.pos for t in env.targets])

    # 计算距离矩阵 (N, M)
    dists = np.linalg.norm(uav_pos[:, None, :] - tgt_pos[None, :, :], axis=2)

    batch_states = []
    metadata = []  # 记录 (u_idx, t_idx) 对应关系

    for i in range(n_uavs):
        uav = env.uavs[i]
        for j in range(n_targets):
            tgt = env.targets[j]

            dist = dists[i, j]
            needed, assigned = tgt.get_demand_status(uav.uav_type)

            # 归一化特征
            feat_dist = dist / 1000.0
            feat_needed = needed / 10.0
            feat_assigned = assigned / 10.0
            feat_satisfied = 1.0 if needed <= 0 else 0.0
            feat_available = 1.0 if uav.available else 0.0

            # 拼装 7维向量
            obs = [feat_dist, 0.0, feat_needed, feat_assigned, feat_satisfied, feat_available, 0.0]

            # 序列化: 复制 5 次
            seq_obs = [obs] * 5

            batch_states.append(seq_obs)
            metadata.append((i, j))

            # 转为 Tensor 并移至 GPU (如果可用)
    device = agent.device
    batch_tensor = torch.FloatTensor(np.array(batch_states)).to(device)  # Shape: (2000, 5, 7)

    # --- 2. GPU/CPU 一次性推理 (The Magic of Batching) ---
    t_start_infer = time.time()

    with torch.no_grad():
        # === 【核心修复部分 Start】 ===
        # 错误写法: logits, _ = agent.policy.actor_net(batch_tensor)
        # 正确流程:
        # 1. 过 Embedding + Transformer 提取特征
        features = agent.policy.actor_net(batch_tensor)  # Shape: (Batch, Seq, Hidden)

        # 2. 取最后一个时间步 (Context Feature)
        context = features[:, -1, :]  # Shape: (Batch, Hidden)

        # 3. 过 MLP Head 得到 Logits
        logits = agent.policy.actor_head(context)  # Shape: (Batch, 2)
        # === 【核心修复部分 End】 ===

        # 计算 Softmax 得到概率
        probs = torch.softmax(logits, dim=1)
        # 取出 "Action 1 (Assign)" 的概率作为得分
        score_assign = probs[:, 1].cpu().numpy()

    t_end_infer = time.time()
    print(f"  [核心] 神经网络推理耗时: {(t_end_infer - t_start_infer):.4f}s (处理 {len(batch_states)} 个配对)")

    # --- 3. 冲突解决 (Conflict Resolution) ---
    candidates = []
    for k, score in enumerate(score_assign):
        u_idx, t_idx = metadata[k]
        tgt = env.targets[t_idx]
        uav = env.uavs[u_idx]

        needed, _ = tgt.get_demand_status(uav.uav_type)
        if needed > 0:
            candidates.append({'score': score, 'u_idx': u_idx, 't_idx': t_idx})

    # 按分数降序排列 (抢票模式: 分高者得)
    candidates.sort(key=lambda x: x['score'], reverse=True)

    # 开始分配
    assignments = []
    uav_occupied = [False] * n_uavs
    temp_demands = {}
    for t in env.targets:
        temp_demands[t.id] = t.demands.copy()

    for item in candidates:
        u_idx = item['u_idx']
        t_idx = item['t_idx']

        uav = env.uavs[u_idx]
        t_id = env.targets[t_idx].id

        if uav_occupied[u_idx]:
            continue

        u_type = uav.uav_type
        if temp_demands[t_id].get(u_type, 0) > 0:
            assignments.append((uav.id, t_id))
            uav_occupied[u_idx] = True
            temp_demands[t_id][u_type] -= 1

    return assignments


def plot_results(env, assignments):
    """简单画图"""
    plt.figure(figsize=(10, 8))
    plt.xlim(0, cfg.MAP_WIDTH)
    plt.ylim(0, cfg.MAP_HEIGHT)
    plt.title(f"Batch Inference Result (Assignments: {len(assignments)})")

    for t in env.targets:
        # 计算该目标当前的总需求
        total_d = sum(t.demands.values())
        plt.scatter(t.pos[0], t.pos[1], c='orange', s=150 + total_d * 20, marker='*', edgecolors='black')
        plt.text(t.pos[0], t.pos[1] + 15, f"T{t.id}\nD:{total_d}", ha='center')

    for u in env.uavs:
        c = 'blue' if any(u.id == x[0] for x in assignments) else 'cyan'
        # alpha=0.1 让未分配的看起来淡一点，突出重点
        alpha = 0.8 if c == 'blue' else 0.1
        plt.scatter(u.pos[0], u.pos[1], c=c, s=50, marker='^', alpha=alpha)

    for u_id, t_id in assignments:
        u = next(x for x in env.uavs if x.id == u_id)
        t = next(x for x in env.targets if x.id == t_id)
        plt.plot([u.pos[0], t.pos[0]], [u.pos[1], t.pos[1]], 'k--', alpha=0.3)

    plt.savefig("batch_vis.png")
    print("结果图已保存至 batch_vis.png")
    plt.show()


if __name__ == "__main__":
    # 1. 加载模型
    model_dir = "./saved_models"
    if not os.path.exists(model_dir):
        print("错误: saved_models 目录不存在")
        exit()

    all_subdirs = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if
                   os.path.isdir(os.path.join(model_dir, d))]
    if not all_subdirs:
        print("错误: 没有找到模型子目录")
        exit()

    latest_subdir = max(all_subdirs, key=os.path.getmtime)

    # 优先找 ep2000，没有找 final，再没有找 best
    possible_names = ["checkpoint_ep2000.pth", "final_model.pth", "best_model.pth"]
    model_path = None
    for name in possible_names:
        p = os.path.join(latest_subdir, name)
        if os.path.exists(p):
            model_path = p
            break

    if model_path is None:
        print(f"错误: 在 {latest_subdir} 中找不到模型文件")
        exit()

    print(f"Loading: {model_path}")

    # 初始化 Agent
    agent = PPOAgent()
    try:
        # weights_only=True 是新版 PyTorch 的安全特性，如果是旧版 PyTorch 可能不支持，需去掉
        ckpt = torch.load(model_path, map_location=agent.device, weights_only=True)
    except Exception as e:
        print(f"尝试加载模型... (Retry without weights_only)")
        ckpt = torch.load(model_path, map_location=agent.device)

    agent.policy.load_state_dict(ckpt)
    agent.policy.eval()


    # 2. 生成大规模场景
    def custom_scenario():
        return {
            "n_uavs": 500,  # <--- 200 架！
            "n_targets": 30,  # <--- 10 个目标
            "n_types": 3,
            "type_ids": [0, 1, 2]
        }


    cfg.generate_scenario = custom_scenario

    env = UAVEnv()
    env.reset()

    print("-" * 60)
    print(f"开始 Batch 推理测试: {len(env.uavs)} UAVs x {len(env.targets)} Targets")
    print("-" * 60)

    start_total = time.time()

    # === 执行 Batch 推理 ===
    matches = batch_inference(env, agent)

    end_total = time.time()

    print("-" * 60)
    print(f"全部完成!")
    print(f"总耗时: {end_total - start_total:.4f} 秒")
    print(f"成功分配: {len(matches)} 对")

    plot_results(env, matches)