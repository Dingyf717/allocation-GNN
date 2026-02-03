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

# ================= 训练参数配置 =================
TRAIN_EPISODES = 100000
PHASE_1_EPS = 10000  # 课程学习第一阶段（简单模式）
TEST_INTERVAL = 100  # 每多少局进行一次评估
TEST_ROUNDS = 20  # 每次评估跑多少局


def save_learning_curve(train_rewards, test_rewards, save_path):
    """
    绘制训练曲线 vs 测试曲线
    """
    plt.figure(figsize=(12, 6))

    # 1. 绘制训练奖励 (带平滑)
    if len(train_rewards) > 0:
        w = max(10, len(train_rewards) // 100)
        ma_train = np.convolve(train_rewards, np.ones(w) / w, mode='valid')
        plt.plot(range(w - 1, len(train_rewards)), ma_train,
                 label=f'Train (Stochastic) MA({w})', color='tab:blue', alpha=0.6)

    # 2. 绘制测试奖励 (贪婪策略)
    # test_rewards 记录的是 (episode_idx, avg_reward) 的元组列表
    if len(test_rewards) > 0:
        x_test = [t[0] for t in test_rewards]
        y_test = [t[1] for t in test_rewards]
        plt.plot(x_test, y_test, label='Test (Greedy)', color='tab:red', linewidth=2)

    plt.title("Training vs Greedy Evaluation")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()


def train():
    print("==================================================")
    print(f"开始训练: Edge-Aware Graph Transformer PPO (Hybrid Mode)")
    print(f"Total Episodes: {TRAIN_EPISODES}")
    print(f"Curriculum Phase 1 (Easy Mode): {PHASE_1_EPS} eps")
    print("==================================================")

    env = UAVEnv()
    agent = PPOAgent()

    # --- 日志系统初始化 ---
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/{time_str}"
    os.makedirs(log_dir, exist_ok=True)
    model_dir = f"./saved_models/{time_str}"
    os.makedirs(model_dir, exist_ok=True)

    # CSV Header 增加了 'Greedy_R'
    csv_file = open(os.path.join(log_dir, "training_stats.csv"), 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["Episode", "Train_Reward", "Greedy_Reward", "L_Actor", "L_Critic", "Entropy", "Mode"])

    # 记录器
    ep_rewards = []  # 训练曲线
    test_log_data = []  # 测试曲线 [(ep, score), ...]

    # 统计变量缓存
    cur_l_act = 0
    cur_l_crt = 0
    cur_ent = 0

    for i_ep in range(1, TRAIN_EPISODES + 1):

        # ================= [Stage 1] 训练模式 (Stochastic) =================
        # 1. 环境重置 & 课程设置
        state = env.reset()

        curriculum_mode = "HARD"
        if i_ep <= PHASE_1_EPS:
            curriculum_mode = "EASY"
            # Phase 1: 强制把所有目标防御设为 1.0 (容易突防)
            for t in env.targets:
                t.defense_level = 1.0

        # 2. 动作选择 (使用 PPO 随机采样进行探索)
        action = agent.select_action(state)

        # 3. 环境步进
        _, reward, done, info = env.step(action)

        # 4. 存储数据
        agent.store_transition(reward, done)
        ep_rewards.append(reward)

        # 5. 模型更新
        if len(agent.buffer['rewards']) >= cfg.BATCH_SIZE:
            stats = agent.update()
            if stats:
                cur_l_act = stats['loss_actor']
                cur_l_crt = stats['loss_critic']
                cur_ent = stats['entropy']

        # ================= [Stage 2] 周期性评估与抽查 =================
        if i_ep % TEST_INTERVAL == 0:
            avg_train_r = np.mean(ep_rewards[-100:])

            # --- A. 执行贪婪验证 (GNN + Greedy Solver) ---
            # 暂停训练，用“GNN+贪婪”模式跑几局，看看真实实力
            greedy_scores = []
            last_test_state = None
            last_test_action = None

            for _ in range(TEST_ROUNDS):
                t_state = env.reset()
                # 再次应用课程模式，保持评估环境与当前训练难度一致
                if curriculum_mode == "EASY":
                    for t in env.targets: t.defense_level = 1.0

                # 【关键】调用 select_greedy_action
                t_action = agent.select_greedy_action(t_state, mode='sequential')

                _, t_r, _, _ = env.step(t_action)
                greedy_scores.append(t_r)

                # 缓存最后一局的数据用于“微观抽查”
                last_test_state = t_state
                last_test_action = t_action

            avg_greedy_r = np.mean(greedy_scores)
            test_log_data.append((i_ep, avg_greedy_r))

            # --- B. 打印宏观日志 ---
            # 比较 Train(随机) 和 Test(贪婪) 的差距
            diff = avg_greedy_r - avg_train_r
            print(f"Ep {i_ep:6d} | Mode: {curriculum_mode} | "
                  f"Train R: {avg_train_r:6.2f} | Test(Greedy) R: {avg_greedy_r:6.2f} ({diff:+.2f}) | "
                  f"Ent: {cur_ent:5.3f}")

            # --- C. 执行战术微观抽查 (Micro-Check) ---
            # 看看第一个无人机到底选了啥，符不符合逻辑
            if last_test_state is not None:
                # 获取 UAV_0 的信息
                # state['uavs'] = (N, 7) -> [x, y, vx, vy, is_d, is_s, is_a]
                u0_info = last_test_state['uavs'][0]
                u0_type_vec = u0_info[-3:]  # 最后3位是one-hot
                u0_type_idx = np.argmax(u0_type_vec)
                type_str = ["Decoy ", "Strike", "Assess"][u0_type_idx]

                # 获取它选的目标
                target_id = last_test_action[0]  # 0是Loiter, 1..M是目标

                print(f"    >>> [战术抽查] UAV_0 [{type_str}] -> ", end="")
                if target_id == 0:
                    print("待命 (Loiter)")
                else:
                    real_t_id = target_id - 1
                    # state['targets'] = (M, 4) -> [x, y, val, def]
                    t_info = last_test_state['targets'][real_t_id]

                    # 计算距离 (需还原归一化坐标)
                    u_pos = u0_info[:2] * cfg.MAP_WIDTH
                    t_pos = t_info[:2] * cfg.MAP_WIDTH
                    dist = np.linalg.norm(u_pos - t_pos)

                    # 防御等级 (归一化是 /10.0，所以还原要 *10)
                    defense = t_info[3] * 10.0

                    print(f"目标_{real_t_id} | 距离: {dist:6.1f}m | 防御: Lv.{defense:.0f}")

            # --- D. 写入文件 ---
            writer.writerow([i_ep, avg_train_r, avg_greedy_r, cur_l_act, cur_l_crt, cur_ent, curriculum_mode])
            csv_file.flush()

        # ================= [Stage 3] 保存模型与曲线 =================
        if i_ep % 5000 == 0:
            torch.save(agent.policy.state_dict(), f"{model_dir}/checkpoint_ep{i_ep}.pth")
            save_learning_curve(ep_rewards, test_log_data, f"{log_dir}/learning_curve.png")

    # 结束
    torch.save(agent.policy.state_dict(), f"{model_dir}/final_model.pth")
    csv_file.close()
    print("训练完成！")


if __name__ == "__main__":
    train()