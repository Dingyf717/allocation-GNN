# test_visualize.py
import os
import time  # 【新增 1】导入 time 模块

# 【新增 2】解决 OpenMP 冲突报错 (必须在 import torch 之前)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
from configs.config import cfg
from envs.uav_env import UAVEnv
from agents.ppo import PPOAgent

# 设置 Matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def visualize_decision(env, agent, model_path):
    print(f"正在加载模型: {model_path} ...")

    # 显式设置 weights_only=True 以消除警告
    # map_location='cpu' 确保兼容性
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)

    agent.policy.load_state_dict(checkpoint)

    # 【关键修复 1】同步权重到 policy_old，防止推理时使用随机参数
    agent.policy_old.load_state_dict(checkpoint)

    agent.policy.eval()
    agent.policy_old.eval()

    state = env.reset()
    done = False
    assignments = []

    print("开始推理决策...")
    print(f"{'决策动作':<30} | {'结果':<8} | {'全队总分 J(X)':<15} | {'本步奖励 Reward'}")
    print("-" * 80)

    # 【新增 3】开始计时
    start_time = time.time()

    while not done:
        # 1. 在 Step 之前获取当前正在做决策的 UAV 和 Target ID
        u_id = env.uavs[env.uav_idx].id
        t_id = env.targets[env.target_idx].id

        # 2. 神经网络决策
        # 【关键修复 2】使用 predict 进行确定性推理 (需先修改 agents/ppo.py)
        # 如果你还没修改 ppo.py，请暂时改回 agent.select_action(state)
        if hasattr(agent, 'predict'):
            action = agent.predict(state)
        else:
            print("警告: PPOAgent 未找到 predict 方法，使用带随机性的 select_action")
            action = agent.select_action(state)

        # 3. 执行环境交互 (获取 info 以读取分数)
        next_state, reward, done, info = env.step(action)

        # 4. 打印决策和分数
        if action == 1:
            # 检查环境是否接受了该分配
            is_valid = info.get('is_valid_action', True)
            status = "✅ 成功" if is_valid else "❌ 拒绝"

            if is_valid:
                assignments.append((u_id, t_id))

            current_j = info.get('J_val', 0.0)
            print(f"UAV-{u_id} 尝试锁定 -> Target-{t_id}    | {status} | {current_j:15.4f} | {reward:+.4f}")

        # 更新状态
        state = next_state

    # 【新增 4】结束计时并计算
    end_time = time.time()
    total_time = end_time - start_time

    # 最终结果打印
    final_j = info.get('J_val', 0.0) if 'info' in locals() else 0.0
    print("-" * 80)
    print(f"决策结束。最终全队总分 J(X): {final_j:.4f}")
    print(f"共生成 {len(assignments)} 个有效攻击对。")
    print(f"算法推理耗时: {total_time:.4f} 秒 (平均每步: {total_time / 300:.4f}s)")  # 假设最大步数约300
    print("-" * 80)

    plot_results(env, assignments)


def plot_results(env, assignments):
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # 1. 画地图边界
    plt.xlim(0, cfg.MAP_WIDTH)
    plt.ylim(0, cfg.MAP_HEIGHT)
    plt.title(f"UAV Swarm Decision Visualization (Map: {cfg.MAP_WIDTH:.0f}x{cfg.MAP_HEIGHT:.0f})")

    # 2. 画禁飞区 (灰色圆圈)
    for nfz in env.nfz_list:
        circle = plt.Circle(nfz.pos, nfz.radius, color='gray', alpha=0.3, label='NFZ')
        ax.add_patch(circle)
        # 画边界
        circle_border = plt.Circle(nfz.pos, nfz.radius, color='black', fill=False, linestyle='--')
        ax.add_patch(circle_border)

    # 3. 画拦截者 (红色X)
    for inter in env.interceptors:
        plt.scatter(inter.pos[0], inter.pos[1], c='red', marker='x', s=100, linewidths=2, label='Interceptor')
        circle = plt.Circle(inter.pos, inter.radius, color='red', alpha=0.1)
        ax.add_patch(circle)

    # 4. 画目标 (根据价值画不同大小的五角星)
    for tgt in env.targets:
        size = tgt.value * 30
        plt.scatter(tgt.pos[0], tgt.pos[1], c='orange', marker='*', s=size, edgecolors='black',
                    label='Target' if tgt.id == 0 else "")
        plt.text(tgt.pos[0], tgt.pos[1] + 2, f"T{tgt.id}\n{tgt.value:.0f}", fontsize=9, ha='center')

    # 5. 画无人机 (蓝色三角)
    for uav in env.uavs:
        plt.scatter(uav.pos[0], uav.pos[1], c='blue', marker='^', s=80, label='UAV' if uav.id == 0 else "")
        plt.text(uav.pos[0], uav.pos[1] - 5, f"U{uav.id}", fontsize=9, ha='center', color='blue')

    # 6. 画连线 (分配关系)
    for (u_id, t_id) in assignments:
        u_pos = env.uavs[u_id].pos
        t_pos = env.targets[t_id].pos
        plt.plot([u_pos[0], t_pos[0]], [u_pos[1], t_pos[1]], 'k--', alpha=0.6)

    # 去重图例
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.grid(True, linestyle=':', alpha=0.6)

    # 保存图片
    save_path = "decision_vis_result.png"
    plt.savefig(save_path, dpi=150)
    print(f"可视化结果已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    # 自动寻找最新的模型
    model_dir = "./saved_models"
    if os.path.exists(model_dir):
        all_subdirs = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if
                       os.path.isdir(os.path.join(model_dir, d))]
        if all_subdirs:
            latest_subdir = max(all_subdirs, key=os.path.getmtime)

            # 优先加载 checkpoint_ep600.pth (根据你的日志这是比较好的模型)
            # 你可以根据需要修改这里的逻辑
            model_path = os.path.join(latest_subdir, "checkpoint_ep2000.pth")

            if not os.path.exists(model_path):
                print(f"提示: {model_path} 不存在，尝试加载 best_model.pth")
                model_path = os.path.join(latest_subdir, "best_model.pth")

            if not os.path.exists(model_path):
                print(f"错误: 找不到模型文件 {model_path}")
            else:
                env = UAVEnv()
                agent = PPOAgent()
                visualize_decision(env, agent, model_path)
        else:
            print(f"错误: {model_dir} 下没有子文件夹")
    else:
        print(f"错误: {model_dir} 文件夹不存在")