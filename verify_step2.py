import numpy as np
from configs.config import cfg
from envs.uav_env import UAVEnv


def verify_logic():
    print("================ 阶段二：核心逻辑验证 ================")

    # 1. 强制初始化一个简单环境
    # 3架UAV: 1 Decoy, 1 Strike, 1 Assess
    # 1个Target: High Value, High Defense
    cfg.MAP_WIDTH = 100.0
    cfg.MAP_HEIGHT = 100.0

    env = UAVEnv()
    env.reset()

    # 手动篡改环境实体，构建“实验室环境”
    print("构造实验场景...")
    env.uavs = []
    # UAV 0: Decoy at (10,10)
    u0 = env._create_dummy_uav(0, [10, 10], cfg.TYPE_DECOY)
    # UAV 1: Strike at (10,10)
    u1 = env._create_dummy_uav(1, [10, 10], cfg.TYPE_STRIKE)
    # UAV 2: Assess at (10,10)
    u2 = env._create_dummy_uav(2, [10, 10], cfg.TYPE_ASSESS)
    env.uavs = [u0, u1, u2]

    env.targets = []
    # Target 0: Value=1.0, Defense=2.0 at (20,20)
    t0 = env._create_dummy_target(0, [20, 20], value=1.0, defense=2.0)
    env.targets = [t0]

    print(f"Target 0: Value={t0.value}, Defense={t0.defense_level}")
    print(f"UAVs: 0=Decoy, 1=Strike, 2=Assess")
    print("-" * 50)

    # --- Case 1: 完美配合 (All -> T0) ---
    # Decoy(1) < Defense(2) -> P_pen = 0.5
    # Strike(1) -> P_dmg = 0.6
    # Assess(1) -> I_info = 1.0
    # Exp Reward = 1.0 * 0.5 * 0.6 * 1.0 * 10.0 = 3.0 (忽略距离)
    action = [1, 1, 1]  # 动作1代表Target 0
    _, r, _, info = env.step(action)
    print(f"Case 1 [完美配合]: Reward={r:.4f} (Mission={info['mission_reward']:.4f})")
    detail = info['details'][0]
    print(f"  -> Pen={detail['p_pen']:.2f}, Dmg={detail['p_dmg']:.2f}, Info={detail['i_info']}")
    assert detail['p_pen'] == 0.5, "❌ 突防概率计算错误"
    assert detail['i_info'] == 1.0, "❌ 信息闭环检测错误"

    # --- Case 2: 缺少评估 (No Assess) ---
    # UAV 2 去 Loiter (Action 0)
    action = [1, 1, 0]
    _, r, _, info = env.step(action)
    print(f"Case 2 [缺少评估]: Reward={r:.4f}")
    detail = info['details'][0]
    assert detail['reward'] == 0.0, "❌ 缺少评估机应该得 0 分"

    # --- Case 3: 缺少打击 (No Strike) ---
    # UAV 1 去 Loiter
    action = [1, 0, 1]
    _, r, _, info = env.step(action)
    print(f"Case 3 [缺少打击]: Reward={r:.4f}")
    detail = info['details'][0]
    assert detail['p_dmg'] == 0.0, "❌ 缺少打击机毁伤应为 0"

    # --- Case 4: 饱和突防 (增加 Decoy) ---
    # 假设我们修改防御为 1.0，此时 1架 Decoy 应该满分
    env.targets[0].defense_level = 1.0
    action = [1, 1, 1]
    _, r, _, info = env.step(action)
    print(f"Case 4 [防御降低]: Reward={r:.4f}")
    detail = info['details'][0]
    assert detail['p_pen'] == 1.0, "❌ 防御降低后突防应该满值"

    print("✅ 阶段二验证通过：核心门控逻辑与公式实现正确。")


# 辅助函数：猴子补丁 (Monkey Patching) 用于给 Env 添加测试用的 helper
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
    verify_logic()