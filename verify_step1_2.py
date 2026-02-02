# verify_step1_2.py
import numpy as np
from configs.config import cfg
from envs.uav_env import UAVEnv


def verify():
    print("=== Step 1 & 2 Verification ===")

    # 1. 检查 Config
    print(f"Checking Config...")
    print(f"UAV_STATE_DIM: {cfg.UAV_STATE_DIM} (Expect 6)")
    print(f"TARGET_STATE_DIM: {cfg.TARGET_STATE_DIM} (Expect 5 if MAX_TYPES=3)")

    # 2. 初始化环境
    env = UAVEnv()

    # 3. 调用 Reset 获取全局数据
    print("\nCalling env.reset()...")
    uav_states, target_states = env.reset()

    # 4. 打印形状
    n_uavs = len(env.uavs)
    n_targets = len(env.targets)

    print(f"\nGenerated Scenario:")
    print(f"  - Num UAVs: {n_uavs}")
    print(f"  - Num Targets: {n_targets}")

    print(f"\nObservation Shapes:")
    print(f"  - UAV Matrix:    {uav_states.shape}  | Expected: ({n_uavs}, {cfg.UAV_STATE_DIM})")
    print(f"  - Target Matrix: {target_states.shape}  | Expected: ({n_targets}, {cfg.TARGET_STATE_DIM})")

    # 5. 验证数值有效性
    if uav_states.shape[1] == cfg.UAV_STATE_DIM and target_states.shape[1] == cfg.TARGET_STATE_DIM:
        print("\n✅ 验证通过！环境已成功改为输出全局状态矩阵。")
    else:
        print("\n❌ 验证失败：维度不匹配。")


if __name__ == "__main__":
    verify()