# verify_step4.py
import numpy as np
from envs.uav_env import UAVEnv
from agents.ppo import PPOAgent


def verify_agent():
    print("=== Step 4 Verification: Agent & Data Flow ===")

    env = UAVEnv()
    agent = PPOAgent()

    # 1. 获取一个状态
    state = env.reset()
    print("State Shapes:", state[0].shape, state[1].shape)

    # 2. Select Action
    action = agent.select_action(state)
    print("\nSelected Action:")
    print(f"  Shape: {action.shape} (Should be N_uav,)")
    print(f"  Example: {action[:10]}")

    # 3. Store Transition
    agent.store_transition(reward=10.0, done=True)

    # 4. Mock Update (Force run)
    # 再存几个 dummy 数据，确保存储不够 BatchSize 也能跑 (代码里有 min 处理)
    for _ in range(5):
        s = env.reset()  # 注意：reset 可能会改变 N_uav，用于测试 Padding
        a = agent.select_action(s)
        agent.store_transition(reward=5.0, done=True)

    print("\nRunning Update (with variable lengths)...")
    try:
        stats = agent.update()
        print("Update Success!")
        print("Stats:", stats)
        print("\n✅ PPO Agent 逻辑验证通过！One-Shot 变长 Batch 跑通。")
    except Exception as e:
        print("\n❌ Update Failed.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    verify_agent()