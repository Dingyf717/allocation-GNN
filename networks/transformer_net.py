# networks/transformer_net.py
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from configs.config import cfg


class DeepAllocationNet(nn.Module):
    """
    Stage 4 (Fixed): Edge-Aware Graph Transformer
    架构: Node Encoder + Edge Encoder -> Edge-Injected Cross-Attention
    修复: Critic 现在也会聚合 Edge 特征，从而感知几何结构(距离/角度)
    """

    def __init__(self):
        super(DeepAllocationNet, self).__init__()

        self.embed_dim = cfg.EMBED_DIM

        # ================= 1. Encoders =================
        # UAV Node Encoder
        self.uav_embedding = nn.Sequential(
            nn.Linear(cfg.UAV_STATE_DIM, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU()
        )
        # Self-Attention (UAVs 协作)
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

        # Edge Encoder
        self.edge_embedding = nn.Sequential(
            nn.Linear(cfg.EDGE_DIM, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU()
        )
        # Edge Bias Projector
        self.edge_gate = nn.Linear(self.embed_dim, 1)

        # ================= 2. Decoder =================
        # Loiter Token
        self.skip_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        self.scale = self.embed_dim ** -0.5

        # ================= 3. Critic =================
        self.critic_head = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def get_action(self, uav_states, target_states, edge_features, action=None):
        """
        参数:
            uav_states: (Batch, N, 7)
            target_states: (Batch, M, 4)
            edge_features: (Batch, N, M, 2)
        """
        batch_size, n_uav, _ = uav_states.shape
        _, n_tgt, _ = target_states.shape

        # 1. Node Encoding
        u_emb = self.uav_embedding(uav_states)  # (B, N, E)
        u_emb = self.uav_encoder(u_emb)  # (B, N, E)

        t_emb = self.target_embedding(target_states)  # (B, M, E)

        # 2. Edge Encoding & Bias
        # e_emb: (B, N, M, E)
        e_emb = self.edge_embedding(edge_features)

        # Calculate Bias for Actor: (B, N, M)
        edge_bias = self.edge_gate(e_emb).squeeze(-1)

        # 3. Cross Attention with Edge Injection (Actor Logic)
        # Part A: UAVs -> Real Targets
        logits_tgt = torch.matmul(u_emb, t_emb.transpose(1, 2)) * self.scale
        # 【核心】注入物理约束
        logits_tgt = logits_tgt + edge_bias

        # Part B: UAVs -> Loiter
        skip_emb = self.skip_token.expand(batch_size, 1, -1)
        logits_skip = torch.matmul(u_emb, skip_emb.transpose(1, 2)) * self.scale

        # Output Logits
        logits = torch.cat([logits_skip, logits_tgt], dim=2)

        # 4. Action Distribution
        logits_flat = logits.reshape(-1, n_tgt + 1)
        dist = Categorical(logits=logits_flat)

        if action is None:
            action_flat = dist.sample()
            action = action_flat.view(batch_size, n_uav)
        else:
            action_flat = action.view(-1)

        log_prob_flat = dist.log_prob(action_flat)
        entropy_flat = dist.entropy()

        log_prob = log_prob_flat.view(batch_size, n_uav)
        entropy = entropy_flat.view(batch_size, n_uav)

        # 5. Critic Value (Fixed: Geometry-Aware)
        # 聚合 UAV 特征
        g_u = torch.max(u_emb, dim=1)[0]  # (B, E)
        # 聚合 Target 特征
        g_t = torch.max(t_emb, dim=1)[0]  # (B, E)

        # 【修复】聚合 Edge 特征
        # e_emb 是 (B, N, M, E)。我们需要将其展平并池化，以提取"全局几何特征"
        # 例如：如果在 N*M 条边中有一条边特征很好（距离极近），Critic 应该能感知到
        g_e = torch.max(e_emb.flatten(1, 2), dim=1)[0]  # (B, E)

        # 简单融合
        global_feat = g_u + g_t + g_e

        value = self.critic_head(global_feat)  # (B, 1)

        return action, log_prob, value, entropy

    def evaluate(self, uav_states, target_states, edge_features, action):
        _, log_prob, value, entropy = self.get_action(uav_states, target_states, edge_features, action)
        return log_prob, value, entropy