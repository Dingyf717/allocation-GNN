# agents/ppo.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import BatchSampler, SubsetRandomSampler

from configs.config import cfg
from networks.transformer_net import TransformerActorCritic


class PPOAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = TransformerActorCritic().to(self.device)
        self.optimizer = optim.Adam([
            {'params': self.policy.actor_head.parameters(), 'lr': cfg.LR_ACTOR},
            {'params': self.policy.actor_net.parameters(), 'lr': cfg.LR_ACTOR},
            {'params': self.policy.critic_head.parameters(), 'lr': cfg.LR_CRITIC},
            {'params': self.policy.critic_net.parameters(), 'lr': cfg.LR_CRITIC},
        ])

        # # 2. [新增] 定义学习率调度器 (Scheduler)
        # # 方案：StepLR。每过 step_size 个单位，LR = LR * gamma
        # # 这里 step_size 设为 200，意味着每 200 个 Episode 衰减一次
        # # gamma 设为 0.9，意味着每次衰减为原来的 90%
        # # 这样到 Episode 2000 时，LR 约为初始的 0.9^10 ≈ 0.35倍，既能稳住后期，又不会太小导致学不动
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.9)

        self.policy_old = TransformerActorCritic().to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.buffer = {
            'states': [],
            'actions': [],
            'logprobs': [],
            'rewards': [],
            'is_terminals': [],
            'values': []
        }

        self.mse_loss = nn.MSELoss()

    # 3. [新增] 外部调用接口
    # def update_lr(self):
    #     self.scheduler.step()
    #     # 可选：打印当前学习率方便调试
    #     current_lr = self.optimizer.param_groups[0]['lr']
    #     # print(f"Learning Rate Updated: {current_lr:.2e}")

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():

            action, log_prob, value, _ = self.policy_old.get_action(state)

        self.buffer['states'].append(state.cpu())  # Keep on CPU to save GPU memory
        self.buffer['actions'].append(action.cpu())
        self.buffer['logprobs'].append(log_prob.cpu())
        self.buffer['values'].append(value.cpu())

        return action.item()

    def predict(self, state):
        """
        用于测试/可视化的确定性推理 (Deterministic Inference)。
        原理：直接取概率最大的动作 (Argmax)，而不是按概率分布采样 (Sample)。
        优点：
        1. 消除随机性，展示模型的最优决策。
        2. 不会将数据存入 Buffer，避免测试时的内存浪费。
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # 1. Actor forward (直接调用 self.policy，确保用的是加载了权重的模型)
            # 注意：这里的逻辑复刻了 transformer_net.py 中的 get_action 前半部分
            x = self.policy.actor_net(state)

            # 提取序列最后一个时间步的特征 (与 get_action 保持一致)
            x = x[:, -1, :]

            logits = self.policy.actor_head(x)

            # 2. Argmax 选择概率最大的动作
            action = torch.argmax(logits, dim=-1)

        return action.item()

    def store_transition(self, reward, done):
        self.buffer['rewards'].append(reward)
        self.buffer['is_terminals'].append(done)

    def update(self):
        # 1. 准备数据
        rewards = self.buffer['rewards']
        is_terminals = self.buffer['is_terminals']
        values = torch.cat(self.buffer['values'], dim=0).squeeze().to(self.device)

        # GAE Calculation
        # 因为我们是在 Episode 结束后更新，所以最后一步的 next_value 肯定是 0
        # 构造 next_values 用于 GAE
        next_values = torch.cat([values[1:], torch.tensor([0.0], device=self.device)])

        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            r_t = rewards[step]
            done = is_terminals[step]
            v_t = values[step]
            v_next = next_values[step] if not done else 0.0

            delta = r_t + cfg.GAMMA * v_next - v_t
            gae = delta + cfg.GAMMA * cfg.GAE_LAMBDA * gae * (1.0 - float(done))
            returns.insert(0, gae + v_t)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = returns - values
        # Advantage Normalization
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # 转换 Buffer 为 Tensor
        old_states = torch.cat(self.buffer['states'], dim=0).to(self.device)
        old_actions = torch.cat(self.buffer['actions'], dim=0).to(self.device)
        old_logprobs = torch.cat(self.buffer['logprobs'], dim=0).to(self.device)
        old_values = values.detach()

        # 记录累加值
        sum_loss_actor = 0
        sum_loss_critic = 0
        sum_entropy = 0
        update_count = 0

        # 2. PPO 更新循环 (Mini-Batch Update)
        dataset_size = len(old_states)
        batch_size = cfg.BATCH_SIZE

        for _ in range(cfg.K_EPOCHS):
            # 使用 BatchSampler 生成 Mini-batches
            # sampler = BatchSampler(SubsetRandomSampler(range(dataset_size)), batch_size, drop_last=False)
            sampler = BatchSampler(SubsetRandomSampler(range(dataset_size)), batch_size, drop_last=True)

            for indices in sampler:
                indices = torch.tensor(indices, device=self.device)

                # 获取 Mini-batch 数据
                batch_states = old_states[indices]
                batch_actions = old_actions[indices]
                batch_old_logprobs = old_logprobs[indices]
                batch_old_values = old_values[indices]
                batch_advantages = advantages[indices]
                batch_returns = returns[indices]

                # Evaluate
                logprobs, state_values, dist_entropy = self.policy.evaluate(batch_states, batch_actions)
                state_values = torch.squeeze(state_values)

                # Ratio
                ratios = torch.exp(logprobs - batch_old_logprobs)

                # Surrogate Loss
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - cfg.EPS_CLIP, 1 + cfg.EPS_CLIP) * batch_advantages
                loss_actor = -torch.min(surr1, surr2).mean()

                # Critic Loss (with Value Clipping for stability)
                # V_loss_1 = (V - Return)^2
                # V_loss_2 = (V_clipped - Return)^2
                v_pred_clipped = batch_old_values + torch.clamp(state_values - batch_old_values, -cfg.EPS_CLIP,
                                                                cfg.EPS_CLIP)
                loss_v1 = self.mse_loss(state_values, batch_returns)
                loss_v2 = self.mse_loss(v_pred_clipped, batch_returns)
                loss_critic = torch.max(loss_v1, loss_v2)  # 取 max 实际上是取 worst case，有些实现是取 0.5 * max

                # Entropy Bonus
                loss_entropy = -dist_entropy.mean()

                # Total Loss
                loss = loss_actor + 0.5 * loss_critic + 0.01 * loss_entropy

                # Backward
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient Clipping (Table I: threshold=1)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.GRAD_NORM_CLIP)

                # --- 【新增】收集统计值 ---
                sum_loss_actor += loss_actor.item()
                # 注意：critic loss 我们记录原始的 loss_v1 或加权的 loss_critic 都可以
                sum_loss_critic += loss_critic.item()
                sum_entropy += dist_entropy.mean().item()
                update_count += 1

                self.optimizer.step()

        # 3. 更新旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.clear_buffer()

        # --- 【修改】返回平均 Loss ---
        if update_count == 0: return None
        return {
            "loss_actor": sum_loss_actor / update_count,
            "loss_critic": sum_loss_critic / update_count,
            "entropy": sum_entropy / update_count
        }

    def clear_buffer(self):
        self.buffer = {k: [] for k in self.buffer}