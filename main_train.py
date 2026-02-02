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

# ================= 配置覆盖 =================
# 强制覆盖 Config 中的 MAX_EPISODES，因为 One-Shot 需要更多局数
# 建议至少跑 50,000 ~ 100,000 局 (相当于以前的 1000~2000 局数据量)
TRAIN_EPISODES = 100000


# ===========================================

def save_learning_curve(rewards, save_path):
    plt.figure(figsize=(12, 6))
    # 数据太多，只画平滑曲线，不画原始噪点
    if len(rewards) > 0:
        # window size
        w = max(10, len(rewards) // 100)
        ma = np.convolve(rewards, np.ones(w) / w, mode='valid')
        plt.plot(range(w - 1, len(rewards)), ma, label=f'Moving Avg ({w})', color='tab:orange', linewidth=2)

    plt.title(f"Learning Curve: Deep Scheme (Target: {TRAIN_EPISODES} Eps)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train():
    print("============================================================================================")
    print(f"开始训练 Deep Scheme PPO (Encoder-Decoder)")
    print(f"模式: Global One-Shot | 目标局数: {TRAIN_EPISODES}")
    print("============================================================================================")

    env = UAVEnv()
    agent = PPOAgent()

    # 日志设置
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/{time_str}"
    os.makedirs(log_dir, exist_ok=True)
    model_dir = f"./saved_models/{time_str}"
    os.makedirs(model_dir, exist_ok=True)

    csv_path = os.path.join(log_dir, "training_stats.csv")
    csv_file = open(csv_path, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Episode", "Reward", "Sat_Rate", "Valid_Assigns", "Loss_Actor", "Loss_Critic", "Entropy"])

    ep_rewards = []
    ep_sat_rates = []

    # 用于平滑显示的变量
    running_loss_actor = 0.0
    running_loss_critic = 0.0
    running_entropy = 0.0

    # ================= 主循环 =================
    for i_episode in range(1, TRAIN_EPISODES + 1):

        # 1. Reset (获取全局状态矩阵)
        state = env.reset(full_reset=True)

        # 2. Action (One-Shot 全局决策)
        action = agent.select_action(state)

        # 3. Step (一次性结算)
        _, reward, done, info = env.step(action)

        # 4. Store
        agent.store_transition(reward, done)

        # 5. Logging Vars
        ep_rewards.append(reward)
        ep_sat_rates.append(info['sat_rate'])

        # 6. Update PPO
        # 只要 Buffer 够了就更新
        if len(agent.buffer['rewards']) >= cfg.BATCH_SIZE:
            ppo_stats = agent.update()
            if ppo_stats:
                # 更新显示变量
                running_loss_actor = ppo_stats['loss_actor']
                running_loss_critic = ppo_stats['loss_critic']
                running_entropy = ppo_stats['entropy']

        # 7. Print & Save (频率降低，避免刷屏)
        # 每 100 局打印一次
        if i_episode % 100 == 0:
            avg_r = np.mean(ep_rewards[-100:])
            avg_sat = np.mean(ep_sat_rates[-100:])

            print(f"Ep {i_episode:6d} | R: {avg_r:8.2f} | Sat: {avg_sat * 100:5.1f}% | "
                  f"L_Act: {running_loss_actor:6.3f} | L_Crt: {running_loss_critic:6.3f} | Ent: {running_entropy:5.3f}")

            csv_writer.writerow(
                [i_episode, avg_r, avg_sat, info['valid_assigns'], running_loss_actor, running_loss_critic,
                 running_entropy])
            csv_file.flush()

        # Checkpoint (每 5000 局保存)
        if i_episode % 5000 == 0:
            torch.save(agent.policy.state_dict(), f"{model_dir}/checkpoint_ep{i_episode}.pth")

        # Plot (每 2000 局画图)
        if i_episode % 2000 == 0:
            save_learning_curve(ep_rewards, f"{log_dir}/learning_curve.png")

    # Finish
    torch.save(agent.policy.state_dict(), f"{model_dir}/final_model.pth")
    csv_file.close()
    print("训练结束！")


if __name__ == "__main__":
    train()