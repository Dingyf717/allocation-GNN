import numpy as np
from configs.config import cfg
from envs.uav_env import UAVEnv


def verify_graph_structure():
    print("================ 阶段三：图观测结构验证 ================")

    # 1. 初始化
    env = UAVEnv()
    obs = env.reset()

    # 2. 检查基本类型
    print(f"Observation Type: {type(obs)}")
    assert isinstance(obs, dict), "❌ Obs 必须是字典"
    assert 'uavs' in obs and 'targets' in obs and 'edges' in obs, "❌ 缺少必要的 Key (uavs, targets, edges)"

    n_u = len(env.uavs)
    n_t = len(env.targets)

    # 3. 检查形状
    print(f"UAV Shape:    {obs['uavs'].shape}  Expect: ({n_u}, 7)")
    print(f"Target Shape: {obs['targets'].shape}  Expect: ({n_t}, 4)")
    print(f"Edge Shape:   {obs['edges'].shape} Expect: ({n_u}, {n_t}, 2)")

    assert obs['edges'].shape == (n_u, n_t, 2), "❌ 边特征维度错误"

    # 4. 验证数值逻辑 (手动构造场景)
    print("\n[数值逻辑验证]")

    # 清空环境，手动放入实体
    env.uavs = []
    env.targets = []

    # Target at (100, 100)
    tgt = env._create_dummy_target(0, [100, 100], 1.0, 1.0)
    env.targets.append(tgt)

    # Case A: UAV at (0, 0), Velocity (1, 1) -> 正对着飞
    # Vector to Target = (100, 100)
    # Velocity = (1, 1)
    # Angle = 0 deg -> Cos = 1.0
    u1 = env._create_dummy_uav(0, [0, 0], cfg.TYPE_STRIKE)
    u1.velocity = np.array([10.0, 10.0])  # 朝东北飞
    env.uavs.append(u1)

    # Case B: UAV at (200, 100), Velocity (1, 0) -> 背对飞 (向东飞，目标在西)
    # Vector to Target = (-100, 0)
    # Velocity = (1, 0)
    # Angle = 180 deg -> Cos = -1.0
    u2 = env._create_dummy_uav(1, [200, 100], cfg.TYPE_STRIKE)
    u2.velocity = np.array([10.0, 0.0])
    env.uavs.append(u2)

    # 重新获取观测
    obs = env._get_obs()
    edges = obs['edges']

    # Check Case A (UAV 0 -> Target 0)
    dist_norm = edges[0, 0, 0]
    cos_val = edges[0, 0, 1]
    print(f"Case A (正对): DistNorm={dist_norm:.3f}, Cos={cos_val:.3f}")
    assert cos_val > 0.99, f"❌ 正对飞行 Cos 应该接近 1.0，实际 {cos_val}"

    # Check Case B (UAV 1 -> Target 0)
    cos_val_b = edges[1, 0, 1]
    print(f"Case B (背对): DistNorm={edges[1, 0, 0]:.3f}, Cos={cos_val_b:.3f}")
    assert cos_val_b < -0.99, f"❌ 背对飞行 Cos 应该接近 -1.0，实际 {cos_val_b}"

    print("✅ 阶段三验证通过：图结构与边特征计算正确。")


# Monkey Patch helper (复用 Stage 2 的 helper)
def _create_dummy_uav(self, id, pos, type_id):
    from envs.entities import UAV
    u = UAV(id, np.array(pos))
    u.reset(np.array(pos), np.zeros(2), type_id)
    return u


def _create_dummy_target(self, id, pos, value, defense):
    from envs.entities import Target
    t = Target(id, np.array(pos))
    t.reset(value, defense)
    return t


UAVEnv._create_dummy_uav = _create_dummy_uav
UAVEnv._create_dummy_target = _create_dummy_target

if __name__ == "__main__":
    verify_graph_structure()