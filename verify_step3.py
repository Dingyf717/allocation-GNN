import torch
from configs.config import cfg
from networks.transformer_net import DeepAllocationNet


def verify_net():
    print("=== Step 3 Verification: Network Architecture ===")

    # 模拟输入数据 (Batch=2, UAV=5, Target=3)
    B, N, M = 2, 5, 3
    u_dim = cfg.UAV_STATE_DIM
    t_dim = cfg.TARGET_STATE_DIM

    dummy_uav = torch.randn(B, N, u_dim)
    dummy_tgt = torch.randn(B, M, t_dim)

    print(f"Input Shapes:")
    print(f"  UAVs: {dummy_uav.shape}")
    print(f"  Targets: {dummy_tgt.shape}")

    # 初始化网络
    net = DeepAllocationNet()
    print("\nInitializing DeepAllocationNet... Done.")

    # 前向传播测试
    action, log_prob, value, entropy = net.get_action(dummy_uav, dummy_tgt)

    print("\nForward Pass Results:")
    print(f"  Action Shape:   {action.shape}   | Expected: ({B}, {N})")
    print(f"  LogProb Shape:  {log_prob.shape}  | Expected: ({B}, {N})")
    print(f"  Value Shape:    {value.shape}    | Expected: ({B}, 1)")

    # 检查 Action 范围
    max_act = action.max().item()
    print(f"  Max Action Idx: {max_act}       | Expected: <= {M} (Indices 0~{M - 1}=Target, {M}=Skip)")

    if action.shape == (B, N) and value.shape == (B, 1):
        print("\n✅ 网络结构验证通过！")
    else:
        print("\n❌ 维度错误，请检查代码。")


if __name__ == "__main__":
    verify_net()