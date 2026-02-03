# agents/ppo.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import BatchSampler, SubsetRandomSampler
import numpy as np

from configs.config import cfg
from networks.transformer_net import DeepAllocationNet


class PPOAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = DeepAllocationNet().to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.LR_ACTOR)

        self.policy_old = DeepAllocationNet().to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Buffer 增加 edge_features
        self.buffer = {
            'uav_states': [],  # List of (N, 7)
            'target_states': [],  # List of (M, 4)
            'edge_features': [],  # List of (N, M, 2)
            'actions': [],  # List of (N,)
            'logprobs': [],
            'rewards': [],
            'is_terminals': [],
            'values': []
        }

        self.mse_loss = nn.MSELoss()

    def select_greedy_action(self, state, mode='sequential'):
        """
        GNN 输出 -> 贪婪决策
        mode: 'simple' (直接取最大) | 'sequential' (带约束的序列分配)
        """
        # 1. 准备数据
        uav_t = torch.FloatTensor(state['uavs']).unsqueeze(0).to(self.device)
        tgt_t = torch.FloatTensor(state['targets']).unsqueeze(0).to(self.device)
        edge_t = torch.FloatTensor(state['edges']).unsqueeze(0).to(self.device)

        # 2. 获取 GNN 评分 (1, N, M+1)
        with torch.no_grad():
            logits = self.policy.get_logits(uav_t, tgt_t, edge_t)
            logits = logits.squeeze(0)  # (N_UAV, M_Target+1)

        n_uav, n_total_opts = logits.shape
        n_targets = n_total_opts - 1

        # 获取 UAV 类型 (用于约束判断)
        # state['uavs'] 是 (N, 7)，最后3位是one-hot类型
        # [..., is_decoy, is_strike, is_assess]
        uav_types = np.argmax(state['uavs'][:, -3:], axis=1)  # 0:Decoy, 1:Strike, 2:Assess

        if mode == 'simple':
            # --- 方案 A: 简单贪婪 (每个 UAV 选自己最喜欢的) ---
            actions = torch.argmax(logits, dim=1).cpu().numpy()
            return actions

        elif mode == 'sequential':
            # --- 方案 B: 带约束的序列贪婪 (推荐) ---
            actions = np.zeros(n_uav, dtype=int)

            # 记录每个目标已分配的各兵种数量
            # allocations[tgt_id][type_id] = count
            allocations = {j: {0: 0, 1: 0, 2: 0} for j in range(n_targets)}

            # 设定约束阈值 (根据你的公式计算得出)
            # 例如: 每个目标最多 5个诱饵, 3个打击, 1个评估
            MAX_CAPACITY = {
                0: 5,  # Decoy Cap
                1: 3,  # Strike Cap
                2: 1  # Assess Cap
            }

            # 这里的 mask 是动态的，用来屏蔽已饱和的目标
            # logits 维度: [N_UAV, 1 + N_Targets] (索引0是Loiter)

            # 为了更好的效果，我们可以先让“更有把握”的 UAV 先选
            # 计算每个 UAV 对其最优目标的“自信度” (Max Logit)
            confidence, _ = torch.max(logits, dim=1)
            sorted_uav_indices = torch.argsort(confidence, descending=True).cpu().numpy()

            for i in sorted_uav_indices:
                u_type = uav_types[i]
                current_scores = logits[i].clone()  # (M+1, )

                # 检查所有目标，如果某目标对当前兵种已饱和，则屏蔽该目标
                for t_id in range(n_targets):
                    if allocations[t_id][u_type] >= MAX_CAPACITY[u_type]:
                        # 屏蔽: 把分数设为极小值 (注意 t_id+1 因为索引0是Loiter)
                        current_scores[t_id + 1] = -1e9

                # 做出选择
                choice = torch.argmax(current_scores).item()
                actions[i] = choice

                # 如果选的不是 Loiter (0)，则更新计数
                if choice > 0:
                    real_tgt_id = choice - 1
                    allocations[real_tgt_id][u_type] += 1

            return actions

    def select_action(self, state):
        """
        输入 state 是 Dict: {'uavs':..., 'targets':..., 'edges':...}
        """
        # 转 Tensor 并增加 Batch 维度 (1, ...)
        uav_t = torch.FloatTensor(state['uavs']).unsqueeze(0).to(self.device)
        tgt_t = torch.FloatTensor(state['targets']).unsqueeze(0).to(self.device)
        edge_t = torch.FloatTensor(state['edges']).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 传入三个参数
            action, log_prob, value, _ = self.policy_old.get_action(uav_t, tgt_t, edge_t)

        # 存入 Buffer (存 CPU 张量)
        self.buffer['uav_states'].append(torch.FloatTensor(state['uavs']))
        self.buffer['target_states'].append(torch.FloatTensor(state['targets']))
        self.buffer['edge_features'].append(torch.FloatTensor(state['edges']))

        self.buffer['actions'].append(action.squeeze(0).cpu())
        self.buffer['logprobs'].append(log_prob.squeeze(0).cpu())
        self.buffer['values'].append(value.squeeze(0).cpu())

        return action.squeeze(0).cpu().numpy()

    def store_transition(self, reward, done):
        self.buffer['rewards'].append(reward)
        self.buffer['is_terminals'].append(done)

    def _collate_batch(self, indices):
        """
        【关键】手动处理变长 3D 张量的 Padding
        """
        # 1. 提取数据
        b_uavs = [self.buffer['uav_states'][i] for i in indices]
        b_tgts = [self.buffer['target_states'][i] for i in indices]
        b_edges = [self.buffer['edge_features'][i] for i in indices]
        b_actions = [self.buffer['actions'][i] for i in indices]
        b_logprobs = [self.buffer['logprobs'][i] for i in indices]

        # 2. 计算最大维度
        batch_size = len(indices)
        max_n = max([u.shape[0] for u in b_uavs])
        max_m = max([t.shape[0] for t in b_tgts])

        # 3. 初始化全零 Padding 张量
        # UAV: (B, Max_N, 7)
        pad_uav = torch.zeros(batch_size, max_n, cfg.UAV_STATE_DIM).to(self.device)
        # Tgt: (B, Max_M, 4)
        pad_tgt = torch.zeros(batch_size, max_m, cfg.TARGET_STATE_DIM).to(self.device)
        # Edge: (B, Max_N, Max_M, 2)
        pad_edge = torch.zeros(batch_size, max_n, max_m, cfg.EDGE_DIM).to(self.device)

        # Actions: (B, Max_N) -> 用 -1 填充，表示无效动作
        pad_act = torch.full((batch_size, max_n), -1, dtype=torch.long).to(self.device)
        pad_logprob = torch.zeros(batch_size, max_n).to(self.device)

        # 4. 填入数据
        for i in range(batch_size):
            cur_n = b_uavs[i].shape[0]
            cur_m = b_tgts[i].shape[0]

            pad_uav[i, :cur_n, :] = b_uavs[i]
            pad_tgt[i, :cur_m, :] = b_tgts[i]
            pad_edge[i, :cur_n, :cur_m, :] = b_edges[i]
            pad_act[i, :cur_n] = b_actions[i]
            pad_logprob[i, :cur_n] = b_logprobs[i]

        return pad_uav, pad_tgt, pad_edge, pad_act, pad_logprob

    def update(self):
        rewards = self.buffer['rewards']
        values = torch.cat(self.buffer['values'], dim=0).to(self.device)

        # --- GAE (One-Shot 简化版) ---
        # 因为 step=1 就结束，returns 就是 rewards
        returns = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        dataset_size = len(rewards)
        batch_size = min(cfg.BATCH_SIZE, dataset_size)

        stats = {'loss_actor': 0, 'loss_critic': 0, 'entropy': 0}

        for _ in range(cfg.K_EPOCHS):
            sampler = BatchSampler(SubsetRandomSampler(range(dataset_size)), batch_size, drop_last=False)

            for indices in sampler:
                # 使用自定义 Collate 函数获取 Padded Batch
                b_uav, b_tgt, b_edge, b_act, b_old_logprob = self._collate_batch(indices)

                b_adv = advantages[indices]
                b_ret = returns[indices]

                # 处理 Action 里的 -1 (Padding)
                # 替换为 0 以防 gather 报错 (计算 Loss 时会被 Mask 掉)
                b_act_safe = b_act.clone()
                b_act_safe[b_act == -1] = 0

                # Forward New Policy
                # 传入 Edge Features
                logprobs, state_values, dist_entropy = self.policy.evaluate(b_uav, b_tgt, b_edge, b_act_safe)

                # Masking (只计算真实 UAV 的 Loss)
                mask = (b_act != -1).float()

                # Joint LogProb Sum
                action_logprobs_sum = (logprobs * mask).sum(dim=1)
                old_logprobs_sum = (b_old_logprob * mask).sum(dim=1)

                ratios = torch.exp(action_logprobs_sum - old_logprobs_sum)

                surr1 = ratios * b_adv
                surr2 = torch.clamp(ratios, 1 - cfg.EPS_CLIP, 1 + cfg.EPS_CLIP) * b_adv
                loss_actor = -torch.min(surr1, surr2).mean()

                loss_critic = self.mse_loss(state_values.squeeze(), b_ret)

                mean_entropy = (dist_entropy * mask).sum() / (mask.sum() + 1e-9)

                loss = loss_actor + 0.5 * loss_critic - 0.01 * mean_entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.GRAD_NORM_CLIP)
                self.optimizer.step()

                stats['loss_actor'] += loss_actor.item()
                stats['loss_critic'] += loss_critic.item()
                stats['entropy'] += mean_entropy.item()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.clear_buffer()

        # Normalize stats
        n_updates = cfg.K_EPOCHS * (dataset_size // batch_size + 1)
        return {k: v / n_updates for k, v in stats.items()}

    def clear_buffer(self):
        for k in self.buffer:
            self.buffer[k] = []