import torch
import numpy as np
from configs.config import cfg
from networks.transformer_net import DeepAllocationNet


def verify_network_upgrade():
    print("================ 阶段四：神经网络升级验证 ================")

    # 1. 实例化网络
    net = DeepAllocationNet()
    print("✅ 网络实例化成功")
    print(f"   Embed Dim: {net.embed_dim}")
    print(f"   Input Dims: UAV={cfg.UAV_STATE_DIM}, Tgt={cfg.TARGET_STATE_DIM}, Edge={cfg.EDGE_DIM}")

    # 2. 构造 Mock 数据 (Batch=2, N=10, M=5)
    B, N, M = 2, 10, 5
    print(f"\n[Input Shapes] Batch={B}, N_uav={N}, N_tgt={M}")

    uav_states = torch.randn(B, N, cfg.UAV_STATE_DIM)
    tgt_states = torch.randn(B, M, cfg.TARGET_STATE_DIM)
    edge_feats = torch.randn(B, N, M, cfg.EDGE_DIM)

    print(f"   UAV Tensor: {uav_states.shape}")
    print(f"   Tgt Tensor: {tgt_states.shape}")
    print(f"   Edge Tensor:{edge_feats.shape}")

    # 3. 前向传播测试
    try:
        action, log_prob, value, entropy = net.get_action(uav_states, tgt_states, edge_feats)

        print("\n[Output Shapes]")
        print(f"   Action:   {action.shape}   Expect: ({B}, {N})")
        print(f"   LogProb:  {log_prob.shape}  Expect: ({B}, {N})")
        print(f"   Value:    {value.shape}     Expect: ({B}, 1)")
        print(f"   Entropy:  {entropy.shape}   Expect: ({B}, {N})")

        # 4. 验证动作范围
        # 动作应该是 0 ~ M (共 M+1 个选项)
        max_act = action.max().item()
        min_act = action.min().item()
        print(f"\n[Logic Check]")
        print(f"   Max Action Index: {max_act} (Should be <= {M})")
        print(f"   Min Action Index: {min_act} (Should be >= 0)")

        if max_act <= M and min_act >= 0:
            print("✅ 动作索引范围正确")
        else:
            print("❌ 动作索引越界")

        # 5. 验证 Edge Injection (梯度检查)
        # 我们需要确认 edge_feats 参与了计算（即有梯度）
        edge_feats.requires_grad = True
        _, _, v, _ = net.get_action(uav_states, tgt_states, edge_feats)
        v.mean().backward()
        if edge_feats.grad is not None and edge_feats.grad.abs().sum() > 0:
            print("✅ Edge Features 梯度回传正常 (物理约束已生效)")
        else:
            print("❌ Edge Features 梯度为零 (物理约束未接入)")

    except Exception as e:
        print(f"\n❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    verify_network_upgrade()