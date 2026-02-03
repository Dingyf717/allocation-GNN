# networks/transformer_net.py
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from configs.config import cfg


class DeepAllocationNet(nn.Module):
    """
    Stage 5 (Refactored): Edge-Aware Graph Transformer
    架构: Node Encoder + Edge Encoder -> Edge-Injected Cross-Attention
    改进:
    1. 解耦 Logits 计算与动作采样，支持贪婪策略 (Greedy Strategy)。
    2. Critic 显式聚合全局 Edge 特征，增强对空间结构的感知。
    """

    def __init__(self):
        super(DeepAllocationNet, self).__init__()

        self.embed_dim = cfg.EMBED_DIM

        # ================= 1. Encoders (特征提取层) =================
        # UAV Node Encoder
        self.uav_embedding = nn.Sequential(
            nn.Linear(cfg.UAV_STATE_DIM, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU()
        )
        # Self-Attention (UAVs 之间的协作通讯)
        self.uav_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=cfg.NUM_HEADS, batch_first=True),
            num_layers=cfg.NUM_LAYERS
        )

        # Target Node Encoder
        self.target_embedding = nn.Sequential(
            nn.Linear(cfg.TARGET_STATE_DIM, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU()
        )

        # Edge Encoder (处理距离、角度等边特征)
        self.edge_embedding = nn.Sequential(
            nn.Linear(cfg.EDGE_DIM, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU()
        )
        # Edge Bias Projector (将边特征投影为 Attention Bias)
        self.edge_gate = nn.Linear(self.embed_dim, 1)

        # ================= 2. Decoder (决策层) =================
        # Loiter Token (代表"不分配/待命"的虚拟目标)
        self.skip_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        # Attention Scale Factor
        self.scale = self.embed_dim ** -0.5

        # ================= 3. Critic (价值评估层) =================
        # 输入融合了 UAV、Target 和 Edge 的全局特征
        self.critic_head = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def _encode(self, uav_states, target_states, edge_features):
        """
        【公共内部方法】统一执行特征编码
        供 get_logits (贪婪决策) 和 get_action (PPO训练) 共同调用，避免代码重复。
        """
        # 1. Node Encoding
        u_emb = self.uav_embedding(uav_states)  # (B, N, E)
        u_emb = self.uav_encoder(u_emb)  # (B, N, E) -> 经过 Transformer 交互

        t_emb = self.target_embedding(target_states)  # (B, M, E)

        # 2. Edge Encoding
        e_emb = self.edge_embedding(edge_features)  # (B, N, M, E)

        # 3. Calculate Bias for Attention
        # 将高维边特征投影为标量 Bias，用于干预 Attention Score
        edge_bias = self.edge_gate(e_emb).squeeze(-1)  # (B, N, M)

        return u_emb, t_emb, e_emb, edge_bias

    def get_logits(self, uav_states, target_states, edge_features):
        """
        【新接口】仅计算并返回原始评分矩阵 (Logits)。
        用于测试阶段的贪婪策略 (Argmax) 或约束求解。
        返回: logits (Batch, N_UAV, N_Target + 1)
        """
        batch_size = uav_states.shape[0]

        # 复用编码逻辑
        u_emb, t_emb, _, edge_bias = self._encode(uav_states, target_states, edge_features)

        # --- Cross Attention with Edge Injection ---

        # Part A: UAVs -> Real Targets (考虑物理约束 edge_bias)
        # (B, N, E) x (B, E, M) -> (B, N, M)
        logits_tgt = torch.matmul(u_emb, t_emb.transpose(1, 2)) * self.scale
        logits_tgt = logits_tgt + edge_bias  # 核心: 注入距离/角度偏置

        # Part B: UAVs -> Loiter (虚拟节点，无物理约束)
        skip_emb = self.skip_token.expand(batch_size, 1, -1)
        logits_skip = torch.matmul(u_emb, skip_emb.transpose(1, 2)) * self.scale

        # Concatenate: Index 0 is Loiter, Index 1..M are Targets
        logits = torch.cat([logits_skip, logits_tgt], dim=2)

        return logits

    def get_action(self, uav_states, target_states, edge_features, action=None):
        """
        【PPO接口】执行完整的动作采样流程，并计算 Critic Value。
        """
        batch_size, n_uav, _ = uav_states.shape
        _, n_tgt, _ = target_states.shape

        # 1. 获取所有特征 (需要 e_emb 来计算 Critic)
        u_emb, t_emb, e_emb, edge_bias = self._encode(uav_states, target_states, edge_features)

        # 2. 计算 Actor Logits (手动展开以利用上面的 u_emb/t_emb)
        logits_tgt = torch.matmul(u_emb, t_emb.transpose(1, 2)) * self.scale + edge_bias
        skip_emb = self.skip_token.expand(batch_size, 1, -1)
        logits_skip = torch.matmul(u_emb, skip_emb.transpose(1, 2)) * self.scale

        # (B, N, M+1)
        logits = torch.cat([logits_skip, logits_tgt], dim=2)

        # 3. 构建动作分布
        logits_flat = logits.reshape(-1, n_tgt + 1)
        dist = Categorical(logits=logits_flat)

        # 4. 采样或评估
        if action is None:
            # Training/Rollout 阶段: 随机采样动作
            action_flat = dist.sample()
            action = action_flat.view(batch_size, n_uav)
        else:
            # Update 阶段: 使用 ReplayBuffer 中的历史动作
            action_flat = action.view(-1)

        # 计算 LogProb 和 Entropy
        log_prob_flat = dist.log_prob(action_flat)
        entropy_flat = dist.entropy()

        log_prob = log_prob_flat.view(batch_size, n_uav)
        entropy = entropy_flat.view(batch_size, n_uav)

        # 5. 计算 Critic Value (Geometry-Aware)
        # Global Pooling: 聚合所有节点和边的信息，形成全局态势感知
        g_u = torch.max(u_emb, dim=1)[0]  # Max Pool over UAVs
        g_t = torch.max(t_emb, dim=1)[0]  # Max Pool over Targets
        g_e = torch.max(e_emb.flatten(1, 2), dim=1)[0]  # Max Pool over Edges (感知全局几何优劣)

        global_feat = g_u + g_t + g_e
        value = self.critic_head(global_feat)  # (B, 1)

        return action, log_prob, value, entropy

    def evaluate(self, uav_states, target_states, edge_features, action):
        """
        PPO Update 专用接口，直接调用 get_action 获取评估数据
        """
        _, log_prob, value, entropy = self.get_action(uav_states, target_states, edge_features, action)
        return log_prob, value, entropy