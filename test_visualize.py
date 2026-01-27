# test_visualize.py
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

# 解决 OpenMP 冲突报错 (必须在 import torch 之前设置)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from configs.config import cfg
from envs.uav_env import UAVEnv
from agents.ppo import PPOAgent

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def visualize_decision(env, agent, model_path):
    print(f"正在加载模型: {model_path} ...")

    # 加载模型权重
    # weights_only=True 消除安全警告，map_location='cpu' 确保在无 GPU 环境也能运行
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    except Exception as e:
        print(f"加载模型失败，尝试不使用 weights_only 参数重试... 错误: {e}")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    agent.policy.load_state_dict(checkpoint)
    # 【关键】同步权重到 policy_old，防止推理时用到未初始化的参数
    agent.policy_old.load_state_dict(checkpoint)

    # 切换到评估模式 (关闭 Dropout/BatchNorm 等)
    agent.policy.eval()
    agent.policy_old.eval()

    # ================= ✄ 开始插入代码 =================
    # 【新增】定义您想要的测试规模
    def custom_scenario():
        return {
            "n_uavs": 96,  # 改这里：无人机数量
            "n_targets": 24,  # 改这里：目标数量
            "n_types": 1,  # 改这里：类型数量
            # "type_ids": list(range(3))
            "type_ids": [0]
        }

    # 临时替换 config 中的生成逻辑
    original_func = cfg.generate_scenario  # 备份原方法
    cfg.generate_scenario = custom_scenario  # 偷梁换柱
    # ================= ✄ 插入结束 =================

    # 重置环境
    state = env.reset()

    cfg.generate_scenario = original_func  # 恢复原样，好习惯

    done = False
    assignments = []

    print("开始推理决策...")
    print(f"{'决策动作':<30} | {'结果':<8} | {'全队总分 J(X)':<15} | {'本步奖励'}")
    print("-" * 80)

    start_time = time.time()

    while not done:
        # 获取当前正在做决策的实体 ID
        if env.uav_idx < len(env.uavs):
            u_id = env.uavs[env.uav_idx].id
        else:
            break  # 防止越界

        if env.target_idx < len(env.targets):
            t_id = env.targets[env.target_idx].id
        else:
            break

        # 1. 神经网络决策
        # 优先使用确定性推理 (predict)，如果未实现则使用随机采样 (select_action)
        if hasattr(agent, 'predict'):
            action = agent.predict(state)
        else:
            action = agent.select_action(state)

        # 2. 执行环境交互
        next_state, reward, done, info = env.step(action)

        # 3. 记录结果
        if action == 1:
            # 读取 info 中的有效性标志 (如果没有 info 则默认有效)
            is_valid = info.get('is_valid_action', True)
            status = "✅ 锁定" if is_valid else "❌ 无效"

            if is_valid:
                assignments.append((u_id, t_id))

            current_j = info.get('J_val', 0.0)
            print(
                f"UAV-{u_id} (Type:{env.uavs[u_id].uav_type}) -> Target-{t_id} | {status} | {current_j:15.4f} | {reward:+.2f}")

        # 更新状态
        state = next_state

    end_time = time.time()
    total_time = end_time - start_time

    # 最终结果打印
    print("-" * 80)
    print(f"决策结束。")
    print(f"共生成 {len(assignments)} 个有效分配。")
    print(f"推理耗时: {total_time:.4f} 秒")
    print("-" * 80)

    plot_results(env, assignments)


def plot_results(env, assignments):
    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    # 1. 画地图边界
    plt.xlim(0, cfg.MAP_WIDTH)
    plt.ylim(0, cfg.MAP_HEIGHT)
    plt.title(f"UAV Swarm Allocation Result (Map: {cfg.MAP_WIDTH:.0f}x{cfg.MAP_HEIGHT:.0f})")

    # 2. 画禁飞区 (兼容性处理：如果环境没有 nfz_list 则跳过)
    if hasattr(env, 'nfz_list'):
        for nfz in env.nfz_list:
            circle = plt.Circle(nfz.pos, nfz.radius, color='gray', alpha=0.3, label='NFZ')
            ax.add_patch(circle)
            ax.add_patch(plt.Circle(nfz.pos, nfz.radius, color='black', fill=False, linestyle='--'))

    # 3. 画拦截者 (兼容性处理)
    if hasattr(env, 'interceptors'):
        for inter in env.interceptors:
            plt.scatter(inter.pos[0], inter.pos[1], c='red', marker='x', s=100, linewidths=2, label='Interceptor')
            ax.add_patch(plt.Circle(inter.pos, inter.radius, color='red', alpha=0.1))

    # 4. 画目标 (Target)
    # 逻辑：新版 Target 可能没有 value 属性，改为使用 demands 总和作为大小依据
    for tgt in env.targets:
        if hasattr(tgt, 'demands'):
            # 计算总需求量
            total_demand = sum(tgt.demands.values())
            # 计算当前还缺多少 (初始需求 - 已分配)
            current_filled = sum(tgt.assigned_counts.values())
            remaining = total_demand - current_filled

            # 显示文本：ID, 总需求, 剩余需求
            label_text = f"T{tgt.id}\nD:{total_demand}"
            size = total_demand * 40 + 100  # 基础大小 + 需求系数
        else:
            # 兼容旧版
            val = getattr(tgt, 'value', 1.0)
            label_text = f"T{tgt.id}\nV:{val:.1f}"
            size = val * 50

        plt.scatter(tgt.pos[0], tgt.pos[1], c='orange', marker='*', s=size, edgecolors='black',
                    label='Target' if tgt.id == 0 else "")
        plt.text(tgt.pos[0], tgt.pos[1] + 15, label_text, fontsize=9, ha='center', fontweight='bold')

    # 5. 画无人机 (UAV)
    for uav in env.uavs:
        # 区分颜色：已分配(实心蓝) vs 未分配(空心/浅色)
        is_assigned = uav.id in [u for u, t in assignments]
        color = 'blue' if is_assigned else 'cyan'

        plt.scatter(uav.pos[0], uav.pos[1], c=color, marker='^', s=80, edgecolors='black',
                    label='UAV' if uav.id == 0 else "")
        # 显示 ID 和 类型
        plt.text(uav.pos[0], uav.pos[1] - 20, f"U{uav.id}\nType:{uav.uav_type}",
                 fontsize=8, ha='center', color='blue')

    # 6. 画连线 (分配关系)
    for (u_id, t_id) in assignments:
        # 查找对象位置 (防止列表乱序)
        u_obj = next((u for u in env.uavs if u.id == u_id), None)
        t_obj = next((t for t in env.targets if t.id == t_id), None)

        if u_obj and t_obj:
            plt.plot([u_obj.pos[0], t_obj.pos[0]], [u_obj.pos[1], t_obj.pos[1]], 'k--', alpha=0.5, linewidth=1)

    # 去重图例
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlabel("X Coordinate (km)")
    plt.ylabel("Y Coordinate (km)")

    # 保存图片
    save_path = "decision_vis_result.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"可视化结果已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    # 自动寻找最新的模型文件夹
    model_dir = "./saved_models"

    if os.path.exists(model_dir):
        # 获取所有子文件夹
        all_subdirs = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if
                       os.path.isdir(os.path.join(model_dir, d))]

        if all_subdirs:
            # 按修改时间排序，找最新的
            latest_subdir = max(all_subdirs, key=os.path.getmtime)
            print(f"检测到最新训练目录: {latest_subdir}")

            # 优先级 1: Final Model
            model_path = os.path.join(latest_subdir, "final_model.pth")

            # 优先级 2: Ep 2000 Checkpoint
            if not os.path.exists(model_path):
                model_path = os.path.join(latest_subdir, "checkpoint_ep2000.pth")

            # 优先级 3: 任何 best_model
            if not os.path.exists(model_path):
                model_path = os.path.join(latest_subdir, "best_model.pth")

            if not os.path.exists(model_path):
                print(f"错误: 在 {latest_subdir} 下找不到可用的模型文件 (.pth)")
            else:
                # 初始化环境和智能体
                env = UAVEnv()
                agent = PPOAgent()

                # 运行可视化
                visualize_decision(env, agent, model_path)
        else:
            print(f"错误: {model_dir} 下没有找到任何训练记录文件夹")
    else:
        print(f"错误: {model_dir} 文件夹不存在，请先运行 main_train.py 进行训练")