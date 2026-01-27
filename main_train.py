# main_train.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
from configs.config import cfg
from envs.uav_env import UAVEnv
from agents.ppo import PPOAgent


def save_learning_curve(rewards, q0s, save_path):
    plt.figure(figsize=(12, 6))

    # 绘制 Reward 曲线 (蓝色)
    plt.plot(rewards, label='Average Reward', color='tab:blue', linewidth=1.5)

    # 绘制 Q0 曲线 (橙色)
    if q0s is not None and len(q0s) > 0:
        plt.plot(q0s, label='Episode Q0 (Value Est.)', color='tab:orange', linewidth=1.5, linestyle='--')

    plt.title("Learning Curve: Reward vs Q0")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train():
    print("============================================================================================")
    print(f"开始训练 PPO-Transformer (泛化架构版)")
    print(f"场景: 动态生成 UAVs 和 Targets | 目标: 供需匹配 + 角度优化")
    print("============================================================================================")

    # 1. 初始化环境和智能体
    env = UAVEnv()
    agent = PPOAgent()

    # 2. 实验记录设置
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/{time_str}"
    os.makedirs(log_dir, exist_ok=True)
    model_dir = f"./saved_models/{time_str}"
    os.makedirs(model_dir, exist_ok=True)

    # --- 【修改】初始化 CSV 日志文件 ---
    csv_path = os.path.join(log_dir, "training_stats.csv")
    csv_file = open(csv_path, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)

    # 【关键修改】更新表头，反映新的业务指标
    csv_writer.writerow([
        "Episode", "Avg_Reward", "Avg_Q0",  # 基础训练指标
        "Avg_J_Value", "Max_Coverage",  # 决策统计
        "Action1_Ratio", "Valid_Assign_Rate",  # 动作分布
        "Sat_Rate", "Avg_Angle",  # 【新指标】满足率 & 角度分
        "Loss_Critic", "Loss_Actor", "Entropy"  # Loss
    ])
    print(f"训练日志将保存至: {csv_path}")

    # 统计变量
    ep_rewards = []
    avg_rewards = []
    ep_q0s = []
    avg_q0s = []

    # 【新增】记录满足率的历史，用于计算平滑曲线
    ep_sat_rates = []

    # ================= 3. 主训练循环 =================
    for i_episode in range(1, cfg.MAX_EPISODES + 1):

        # 重置环境 (full_reset=True 会调用 cfg.generate_scenario 生成新场景)
        state = env.reset(full_reset=True)
        current_ep_reward = 0
        done = False

        # 获取当前回合初始状态的 Value (Q0)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            _, _, q0_val, _ = agent.policy_old.get_action(state_tensor)
            current_q0 = q0_val.item()

        # --- 本回合统计变量 ---
        ep_total_J = 0  # 累计奖励/分数
        ep_steps = 0
        ep_action1_cnt = 0  # 尝试分配次数
        ep_valid_cnt = 0  # 有效分配次数
        ep_total_angle = 0.0  # 累计角度得分

        # --- Episode Loop ---
        while not done:
            # a. select action
            action = agent.select_action(state)
            # b. step
            next_state, reward, done, info = env.step(action)

            # c. store
            agent.store_transition(reward, done)
            # d. update state
            state = next_state
            current_ep_reward += reward

            # --- 收集统计数据 ---
            ep_steps += 1
            if info:
                # 假设 J_val 仍然是某种形式的分数 (在这里通常等于 step reward 的累加或其他定义)
                ep_total_J += info.get('J_val', 0)

                if action == 1:
                    ep_action1_cnt += 1
                    # 检查是否是有效分配 (reward > 0 通常意味着有效，或者由 env 返回标志)
                    if info.get('is_valid_action', False):
                        ep_valid_cnt += 1
                        # 累加角度得分 (需要在 uav_env.py 的 info 中返回 'angle_score')
                        ep_total_angle += info.get('angle_score', 0.0)

        # --- Episode 结束后的 PPO 更新 ---
        ppo_stats = None
        if len(agent.buffer['states']) >= cfg.BATCH_SIZE * 4:
            ppo_stats = agent.update()

        # --- 【关键修改】计算本回合最终的“需求满足率” (Satisfaction Rate) ---
        # 遍历所有 Target，统计已分配 vs 总需求
        current_assigned_sum = 0
        current_demand_sum = 0

        for t in env.targets:
            # 兼容字典结构: t.demands = {0:4, 1:2...}, t.assigned_counts = {0:4, 1:2...}
            t_assigned = sum(t.assigned_counts.values())
            t_remain = sum(t.demands.values())

            current_assigned_sum += t_assigned
            current_demand_sum += (t_remain + t_assigned)  # 总需求 = 剩余 + 已分配

        # 防止除以0
        ep_sat_rate = current_assigned_sum / (current_demand_sum + 1e-6)
        ep_sat_rates.append(ep_sat_rate)

        # 计算平均角度分 (仅针对有效分配)
        avg_angle_score = ep_total_angle / max(1, ep_valid_cnt)

        # --- 记录 Reward 和 Q0 ---
        ep_rewards.append(current_ep_reward)
        avg_reward = np.mean(ep_rewards[-50:])
        avg_rewards.append(avg_reward)

        ep_q0s.append(current_q0)
        avg_q0 = np.mean(ep_q0s[-50:])
        avg_q0s.append(avg_q0)

        # --- 日志输出 (每 10 轮) ---
        if i_episode % 10 == 0:
            # 1. 计算滑动平均值
            avg_J_val = ep_total_J / max(1, ep_steps)
            act1_ratio = ep_action1_cnt / max(1, ep_steps)
            valid_rate = ep_valid_cnt / max(1, ep_action1_cnt)

            # 最近 10 轮的平均满足率
            avg_recent_sat = np.mean(ep_sat_rates[-10:])

            # 获取 Loss
            l_crt = ppo_stats['loss_critic'] if ppo_stats else 0.0
            l_act = ppo_stats['loss_actor'] if ppo_stats else 0.0
            entr = ppo_stats['entropy'] if ppo_stats else 0.0

            # 3. 打印到控制台
            # R: 平均奖励 | Sat%: 需求满足率 | Valid%: 分配有效率 | Ang: 平均角度分
            print(f"Ep {i_episode:4d} | R: {avg_reward:6.2f} | Sat%: {avg_recent_sat * 100:5.1f}% | "
                  f"Valid%: {valid_rate:4.2f} | Ang: {avg_angle_score:.2f} | "
                  f"L_Crt: {l_crt:6.4f} | Ent: {entr:5.3f}")

            # 4. 写入 CSV
            csv_writer.writerow([
                i_episode,
                f"{avg_reward:.4f}", f"{avg_q0:.4f}",
                f"{avg_J_val:.4f}", current_assigned_sum,  # Max_Coverage 记录本轮总分配数
                f"{act1_ratio:.4f}", f"{valid_rate:.4f}",
                f"{avg_recent_sat:.4f}", f"{avg_angle_score:.4f}",  # 新指标写入
                f"{l_crt:.6f}", f"{l_act:.6f}", f"{entr:.6f}"
            ])
            csv_file.flush()

        # 3. 打印简略进度
        if i_episode % 10 == 0:
            # 这里的 avg_reward 和 avg_q0 已经在上面算过了
            pass

            # --- 模型保存逻辑 ---
        if i_episode % 200 == 0:
            ckpt_path = f"{model_dir}/checkpoint_ep{i_episode}.pth"
            torch.save(agent.policy.state_dict(), ckpt_path)
            print(f"   >>> Checkpoint 保存: {ckpt_path}")

        # 定期保存曲线图
        if i_episode % 100 == 0:
            save_learning_curve(avg_rewards, avg_q0s, f"{log_dir}/learning_curve.png")

    # 训练结束
    final_model_path = f"{model_dir}/final_model.pth"
    torch.save(agent.policy.state_dict(), final_model_path)

    csv_file.close()
    print("============================================================================================")
    print("训练结束！")
    print(f"模型已保存至: {model_dir}")
    print(f"日志已保存至: {log_dir}")


if __name__ == "__main__":
    train()