# networks/transformer_net.py
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from configs.config import cfg


def init_layer(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class TransformerBlock(nn.Module):
    """
    Transformer 编码模块
    结构: Linear Embedding -> Transformer Encoder
    """

    def __init__(self, num_layers=cfg.NUM_LAYERS):
        super(TransformerBlock, self).__init__()
        # 1. Embedding 层: 将 State (14维) 映射到 Embed Dim (128维)
        self.embedding = nn.Sequential(
            init_layer(nn.Linear(cfg.STATE_DIM, cfg.EMBED_DIM)),
            nn.ReLU()
        )

        # 位置编码: 建议使用较小的初始化方差
        self.pos_embedding = nn.Parameter(torch.randn(1, cfg.SEQ_LEN, cfg.EMBED_DIM) * 0.02)

        # 2. Transformer Encoder
        # Table I: Heads=8, Channels=128
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.EMBED_DIM,
            nhead=cfg.NUM_HEADS,
            dim_feedforward=256,  # 内部 FFN 维度
            dropout=0.0,
            batch_first=True
        )
        # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.NUM_LAYERS)
        # 使用传入的 num_layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)



    def forward(self, x):
        # x shape: (Batch, Seq_Len, State_Dim)

        # 生成 Padding Mask (防止 Attention 关注到全0的填充部分)
        # mask shape: (Batch, Seq_Len), True 表示需要被忽略
        mask = (x.abs().sum(dim=-1) == 0)

        mask[:, -1] = False

        # Embedding + Positional Encoding
        x_emb = self.embedding(x)
        # 截取对应长度的位置编码 (兼容可能的变长输入，虽然 Config 固定为 5)
        x_emb = x_emb + self.pos_embedding[:, :x.size(1), :]

        # Transformer Forward
        # 注意: src_key_padding_mask 在 PyTorch 中 True 表示忽略
        out = self.transformer(x_emb, src_key_padding_mask=mask)
        return out


class TransformerActorCritic(nn.Module):
    def __init__(self):
        super(TransformerActorCritic, self).__init__()

        # --- 修改点 1: 输入维度不再依赖序列长度，只依赖 Embedding 维度 ---
        # 之前的代码是: self.flat_dim = cfg.SEQ_LEN * cfg.EMBED_DIM
        self.hidden_dim = cfg.EMBED_DIM

        # --- Actor Network ---
        self.actor_net = TransformerBlock(num_layers=1)
        # Head 结构依据 Fig 2: FC -> ReLU -> FC -> Softmax
        self.actor_head = nn.Sequential(
            init_layer(nn.Linear(self.hidden_dim, 64)),  # 输入改为 128 (EMBED_DIM)
            nn.ReLU(),
            init_layer(nn.Linear(64, cfg.ACTION_DIM), std=0.01)
        )

        # --- Critic Network ---
        self.critic_net = TransformerBlock(num_layers=2)
        # Head 结构依据 Fig 2: FC -> ReLU -> FC -> Output
        self.critic_head = nn.Sequential(
            init_layer(nn.Linear(self.hidden_dim, 64)),  # 输入改为 128 (EMBED_DIM)
            nn.ReLU(),
            init_layer(nn.Linear(64, 1), std=1.0)
        )

    def forward(self, state):
        raise NotImplementedError("Please use get_action or evaluate.")

    def get_action(self, state):
        """用于采样动作 (Inference)"""
        if state.dim() == 2: state = state.unsqueeze(0)

        # 1. Actor forward
        x = self.actor_net(state)

        # --- 修改点 2: 提取最后一个时间步的特征 ---
        # x shape: (Batch, Seq_Len, Embed_Dim) -> 取切片 -> (Batch, Embed_Dim)
        # 这一步提取了包含前面所有历史信息的当前时刻上下文特征
        x = x[:, -1, :]

        logits = self.actor_head(x)

        # 2. Critic forward (获取当前状态价值)
        v_x = self.critic_net(state)

        # 同样提取最后一个时间步
        v_x = v_x[:, -1, :]

        value = self.critic_head(v_x)

        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()

        return action, dist.log_prob(action), value, dist.entropy()

    def evaluate(self, state, action):
        """用于 PPO 更新 (Training)"""
        # 1. Actor forward
        x = self.actor_net(state)
        # 提取最后一个时间步
        x = x[:, -1, :]
        logits = self.actor_head(x)

        # 2. Critic forward
        v_x = self.critic_net(state)
        # 提取最后一个时间步
        v_x = v_x[:, -1, :]
        value = self.critic_head(v_x)

        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, value, dist_entropy