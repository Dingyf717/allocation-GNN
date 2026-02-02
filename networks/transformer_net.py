# networks/transformer_net.py
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from configs.config import cfg


class DeepAllocationNet(nn.Module):
    """
    Step 3: Deep Scheme 核心网络
    架构: Encoder (Self-Attention) -> Decoder (Cross-Attention)
    输入: 全局 UAV 状态矩阵, 全局 Target 状态矩阵
    输出: 全局分配概率矩阵 (Batch, N_uav, N_target + 1)
    """

    def __init__(self):
        super(DeepAllocationNet, self).__init__()

        self.embed_dim = cfg.EMBED_DIM

        # ================= 1. Encoders =================
        # UAV 特征提取: Linear -> Transformer Encoder (交互)
        self.uav_embedding = nn.Sequential(
            nn.Linear(cfg.UAV_STATE_DIM, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU()
        )
        # Self-Attention 层: 解决短视问题的核心，让 UAV 互相通信
        self.uav_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=cfg.NUM_HEADS,
            batch_first=True,
            dim_feedforward=256
        )
        self.uav_encoder = nn.TransformerEncoder(self.uav_encoder_layer, num_layers=cfg.NUM_LAYERS)

        # Target 特征提取
        self.target_embedding = nn.Sequential(
            nn.Linear(cfg.TARGET_STATE_DIM, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU()
        )

        # ================= 2. Decoder (Policy Head) =================
        # 这是一个可学习的“Skip”向量，代表“不分配/跳过”这个选项
        # 形状: (1, 1, Embed_Dim)
        self.skip_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        self.scale = self.embed_dim ** -0.5  # 缩放因子

        # ================= 3. Critic Head =================
        # 评估全局状态价值 V(s)
        self.critic_head = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError("Use get_action or evaluate.")

    def _encode_global_state(self, uav_states, target_states):
        """
        前向传播公共部分：编码特征
        """
        # uav_states: (Batch, N, U_dim)
        # target_states: (Batch, M, T_dim)

        # 1. Encode UAVs (With Interaction)
        u_emb = self.uav_embedding(uav_states)  # -> (Batch, N, E)
        u_emb = self.uav_encoder(u_emb)  # -> (Batch, N, E) [Self-Attention happens here]

        # 2. Encode Targets
        t_emb = self.target_embedding(target_states)  # -> (Batch, M, E)

        return u_emb, t_emb

    def get_action(self, uav_states, target_states, action=None):
        """
        计算动作概率与价值
        参数:
            uav_states: (Batch, N_uav, U_dim)
            target_states: (Batch, N_tgt, T_dim)
            action: (Batch, N_uav) Optional, 用于 evaluate 模式
        返回:
            action, log_prob, value, entropy
        """
        # 1. 提取特征
        u_emb, t_emb = self._encode_global_state(uav_states, target_states)
        batch_size, n_uav, _ = u_emb.shape
        batch_size, n_tgt, _ = t_emb.shape

        # 2. 构建 Key 集合 (Targets + Skip Token)
        # 我们把 Skip Token 复制 Batch 份
        skip_emb = self.skip_token.expand(batch_size, 1, -1)  # (B, 1, E)

        # Keys: [Target_0, Target_1, ..., Target_M-1, Skip]
        # Shape: (B, M+1, E)
        keys = torch.cat([t_emb, skip_emb], dim=1)

        # 3. 计算 Logits (Cross-Attention)
        # Query = UAVs (B, N, E)
        # Key   = Targets+Skip (B, M+1, E)
        # Logits = Q * K^T
        # Shape: (B, N, M+1)
        logits = torch.matmul(u_emb, keys.transpose(1, 2)) * self.scale

        # 4. 动作采样 / 概率计算
        # 我们把每个 UAV 视为独立的决策者 (Joint Independent Policy)
        # Flatten 用于构建分布: (B*N, M+1)
        logits_flat = logits.reshape(-1, n_tgt + 1)
        dist = Categorical(logits=logits_flat)

        if action is None:
            # 采样动作
            action_flat = dist.sample()  # (B*N,)
            action = action_flat.view(batch_size, n_uav)
        else:
            # 使用传入的动作 (用于 Update 阶段)
            action_flat = action.view(-1)

        log_prob_flat = dist.log_prob(action_flat)
        entropy_flat = dist.entropy()

        # 还原形状
        log_prob = log_prob_flat.view(batch_size, n_uav)  # (B, N)
        entropy = entropy_flat.view(batch_size, n_uav)  # (B, N)

        # 5. Critic Value Calculation
        # 使用 Global Max Pooling 聚合所有 UAV 和 Target 的信息
        # 这样 Critic 能看到全局信息
        g_u = torch.max(u_emb, dim=1)[0]  # (B, E)
        g_t = torch.max(t_emb, dim=1)[0]  # (B, E)

        global_feat = g_u + g_t  # (B, E)简单融合
        value = self.critic_head(global_feat)  # (B, 1)

        return action, log_prob, value, entropy

    def evaluate(self, uav_states, target_states, action):
        """兼容 PPO update 接口"""
        # 复用 get_action 逻辑，只是不需要返回 action
        _, log_prob, value, entropy = self.get_action(uav_states, target_states, action)
        return log_prob, value, entropy