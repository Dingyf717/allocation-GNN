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

# 训练参数
TRAIN_EPISODES = 100000
PHASE_1_EPS = 10000  # 课程学习第一阶段局数


def save_learning_curve(rewards, save_path):
    plt.figure(figsize=(12, 6))
    if len(rewards) > 0:
        w = max(10, len(rewards) // 100)
        ma = np.convolve(rewards, np.ones(w) / w, mode='valid')
        plt.plot(range(w - 1, len(rewards)), ma, label=f'Mov Avg ({w})', color='tab:orange')
    plt.title("Training Rewards (Curriculum Learning)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()


def train():
    print("==================================================")
    print(f"开始训练: Edge-Aware Graph Transformer PPO")
    print(f"Total Episodes: {TRAIN_EPISODES}")
    print(f"Curriculum Phase 1 (Easy Mode): {PHASE_1_EPS} eps")
    print("==================================================")

    env = UAVEnv()
    agent = PPOAgent()

    # 日志初始化
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/{time_str}"
    os.makedirs(log_dir, exist_ok=True)
    model_dir = f"./saved_models/{time_str}"
    os.makedirs(model_dir, exist_ok=True)

    csv_file = open(os.path.join(log_dir, "training_stats.csv"), 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["Episode", "Reward", "L_Actor", "L_Critic", "Entropy", "Mode"])

    ep_rewards = []

    # 统计变量
    cur_l_act = 0
    cur_l_crt = 0
    cur_ent = 0

    for i_ep in range(1, TRAIN_EPISODES + 1):

        # 1. Reset
        # Curriculum: 在 reset 后修改防御等级
        state = env.reset()

        curriculum_mode = "HARD"
        if i_ep <= PHASE_1_EPS:
            curriculum_mode = "EASY"
            # Phase 1: 强制把所有目标防御设为 1.0 (容易突防)
            for t in env.targets:
                t.defense_level = 1.0

        # 2. Action
        action = agent.select_action(state)

        # 3. Step
        _, reward, done, info = env.step(action)

        # 4. Store
        agent.store_transition(reward, done)
        ep_rewards.append(reward)

        # 5. Update
        if len(agent.buffer['rewards']) >= cfg.BATCH_SIZE:
            stats = agent.update()
            if stats:
                cur_l_act = stats['loss_actor']
                cur_l_crt = stats['loss_critic']
                cur_ent = stats['entropy']

        # 6. Log & Save
        if i_ep % 100 == 0:
            avg_r = np.mean(ep_rewards[-100:])
            print(f"Ep {i_ep:6d} | Mode: {curriculum_mode} | Avg R: {avg_r:7.2f} | "
                  f"L_Act: {cur_l_act:6.3f} | Ent: {cur_ent:5.3f}")

            writer.writerow([i_ep, avg_r, cur_l_act, cur_l_crt, cur_ent, curriculum_mode])
            csv_file.flush()

        if i_ep % 5000 == 0:
            torch.save(agent.policy.state_dict(), f"{model_dir}/checkpoint_ep{i_ep}.pth")
            save_learning_curve(ep_rewards, f"{log_dir}/learning_curve.png")

    torch.save(agent.policy.state_dict(), f"{model_dir}/final_model.pth")
    csv_file.close()
    print("训练完成！")


if __name__ == "__main__":
    train()