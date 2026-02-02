# agents/ppo.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import BatchSampler, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from configs.config import cfg
from networks.transformer_net import DeepAllocationNet


class PPOAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 初始化 Deep Scheme 网络
        self.policy = DeepAllocationNet().to(self.device)

        # 分组优化器参数 (保持与 Config 一致)
        self.optimizer = optim.Adam([
            {'params': self.policy.uav_embedding.parameters(), 'lr': cfg.LR_ACTOR},
            {'params': self.policy.uav_encoder.parameters(), 'lr': cfg.LR_ACTOR},
            {'params': self.policy.target_embedding.parameters(), 'lr': cfg.LR_ACTOR},
            {'params': self.policy.skip_token, 'lr': cfg.LR_ACTOR},
            {'params': self.policy.critic_head.parameters(), 'lr': cfg.LR_CRITIC},
        ])

        self.policy_old = DeepAllocationNet().to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 2. 修改 Buffer 结构: 分离 UAV 和 Target 状态
        self.buffer = {
            'uav_states': [],  # List of Tensor (N, D_u)
            'target_states': [],  # List of Tensor (M, D_t)
            'actions': [],  # List of Tensor (N,)
            'logprobs': [],  # List of Tensor (N,)
            'rewards': [],  # List of float (Global Scalar Reward)
            'is_terminals': [],  # List of bool
            'values': []  # List of Tensor (1,)
        }

        self.mse_loss = nn.MSELoss()

    def select_action(self, state):
        """
        One-Shot 动作选择
        State: Tuple (uav_matrix, target_matrix)
        """
        uav_mat, tgt_mat = state

        # 转 Tensor (增加 Batch 维度 B=1)
        uav_t = torch.FloatTensor(uav_mat).unsqueeze(0).to(self.device)  # (1, N, D_u)
        tgt_t = torch.FloatTensor(tgt_mat).unsqueeze(0).to(self.device)  # (1, M, D_t)

        with torch.no_grad():
            # 获取全局动作
            action, log_prob, value, _ = self.policy_old.get_action(uav_t, tgt_t)

        # 存入 Buffer (移除 Batch 维度，存原始数据)
        self.buffer['uav_states'].append(torch.FloatTensor(uav_mat))
        self.buffer['target_states'].append(torch.FloatTensor(tgt_mat))
        self.buffer['actions'].append(action.squeeze(0).cpu())  # (N,)
        self.buffer['logprobs'].append(log_prob.squeeze(0).cpu())  # (N,)
        self.buffer['values'].append(value.squeeze(0).cpu())  # (1,)

        # 返回 Numpy 动作矩阵
        return action.squeeze(0).cpu().numpy()

    def predict(self, state):
        """
        推理模式 (Deterministic 建议用 argmax，这里暂时复用 get_action 采样)
        后续可优化为输出 Logits 给匈牙利算法
        """
        uav_mat, tgt_mat = state
        uav_t = torch.FloatTensor(uav_mat).unsqueeze(0).to(self.device)
        tgt_t = torch.FloatTensor(tgt_mat).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, _, _, _ = self.policy.get_action(uav_t, tgt_t)

        return action.squeeze(0).cpu().numpy()

    def store_transition(self, reward, done):
        self.buffer['rewards'].append(reward)
        self.buffer['is_terminals'].append(done)

    def update(self):
        # 1. 准备数据
        rewards = self.buffer['rewards']
        # values 是 List of (1,) Tensor
        values = torch.cat(self.buffer['values'], dim=0).to(self.device)  # (Buffer_Size,)

        # --- GAE 计算 (针对 One-Shot 简化版) ---
        returns = []
        tensor_rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        advantages = tensor_rewards - values.detach()
        returns = tensor_rewards

        # Normalize Advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # 2. 数据对齐与填充 (Padding)
        # batch_first=True -> (Batch, Max_Len, Dim)
        old_uav_states = pad_sequence(self.buffer['uav_states'], batch_first=True).to(self.device)
        old_target_states = pad_sequence(self.buffer['target_states'], batch_first=True).to(self.device)
        old_actions = pad_sequence(self.buffer['actions'], batch_first=True, padding_value=-1).to(
            self.device)  # Pad with -1
        old_logprobs = pad_sequence(self.buffer['logprobs'], batch_first=True).to(self.device)

        # 3. PPO Update Loop
        dataset_size = len(rewards)
        batch_size = min(cfg.BATCH_SIZE, dataset_size)

        sum_loss_actor = 0
        sum_loss_critic = 0
        sum_entropy = 0
        update_count = 0

        for _ in range(cfg.K_EPOCHS):
            sampler = BatchSampler(SubsetRandomSampler(range(dataset_size)), batch_size, drop_last=False)

            for indices in sampler:
                indices = torch.tensor(indices, device=self.device)

                # Slice Batch
                b_uav = old_uav_states[indices]
                b_tgt = old_target_states[indices]
                b_act = old_actions[indices]  # (Batch, Max_N), contains -1
                b_old_logprob = old_logprobs[indices]
                b_adv = advantages[indices]
                b_ret = returns[indices]

                # ================= 修复核心 =================
                # PyTorch 的 Categorical 不能处理 -1，即使我们后面会 Mask 掉。
                # 所以我们创建一个 safe 的 action tensor，把 -1 替换成 0。
                # 这样做不会影响结果，因为对应的 Loss 权重会被 mask 设为 0。
                b_act_safe = b_act.clone()
                b_act_safe[b_act == -1] = 0
                # ===========================================

                # Evaluate (使用 safe actions)
                logprobs, state_values, dist_entropy = self.policy.evaluate(b_uav, b_tgt, b_act_safe)

                # State Value: (Batch, 1) -> (Batch,)
                state_values = state_values.squeeze()

                # Masking: 生成掩码，忽略 Padding 部分 (原始 b_act == -1 的位置)
                mask = (b_act != -1).float()

                # 聚合 LogProb: (Batch, Max_N) -> (Batch,)
                # Sum log probability over all agents => Joint probability
                action_logprobs_sum = (logprobs * mask).sum(dim=1)
                old_logprobs_sum = (b_old_logprob * mask).sum(dim=1)

                ratios = torch.exp(action_logprobs_sum - old_logprobs_sum)

                # Surrogate Loss
                surr1 = ratios * b_adv
                surr2 = torch.clamp(ratios, 1 - cfg.EPS_CLIP, 1 + cfg.EPS_CLIP) * b_adv
                loss_actor = -torch.min(surr1, surr2).mean()

                # Critic Loss
                loss_critic = self.mse_loss(state_values, b_ret)

                # Entropy (Masked mean)
                mean_entropy = (dist_entropy * mask).sum() / (mask.sum() + 1e-9)

                loss = loss_actor + 0.5 * loss_critic - 0.01 * mean_entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.GRAD_NORM_CLIP)
                self.optimizer.step()

                sum_loss_actor += loss_actor.item()
                sum_loss_critic += loss_critic.item()
                sum_entropy += mean_entropy.item()
                update_count += 1

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.clear_buffer()

        if update_count == 0: return None
        return {
            "loss_actor": sum_loss_actor / update_count,
            "loss_critic": sum_loss_critic / update_count,
            "entropy": sum_entropy / update_count
        }

    def clear_buffer(self):
        for k in self.buffer:
            self.buffer[k] = []