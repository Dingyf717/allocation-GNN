import numpy as np
from configs.config import cfg
from envs.entities import UAV, Target
from envs.mechanics import calc_angle_score, calc_dist_score, calc_speed_score, calc_damage_prob


def print_separator(title):
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def diagnose_scenario(name, dist_km, angle_deg, uav_speed_km_s=0.4, tgt_speed_km_s=0.01):
    print_separator(f"场景测试: {name}")

    # 1. 构造虚拟实体
    # 假设 UAV 在原点，目标在 (dist, 0)
    uav_pos = np.array([0.0, 0.0])
    target_pos = np.array([float(dist_km), 0.0])

    # UAV 速度方向：与x轴夹角为 angle_deg
    # 目标在正x轴方向，所以 angle_deg 就是 sigma (偏差角)
    theta_rad = np.deg2rad(angle_deg)
    uav_vel = np.array([np.cos(theta_rad), np.sin(theta_rad)]) * uav_speed_km_s

    target_vel = np.array([-1.0, 0.0]) * tgt_speed_km_s  # 目标迎面飞来

    # 创建对象
    uav = UAV(id=0, pos=uav_pos, velocity=uav_vel, max_speed=uav_speed_km_s, load=1.0)
    uav.cost = 1.0  # 假设是 Type 1 UAV
    target = Target(id=0, pos=target_pos, value=4.0)  # 假设是 Level 1 Target

    target.velocity = target_vel

    # 2. 调用 mechanics 进行计算
    # (A) 距离得分
    e_dist = calc_dist_score(dist_km, is_obstacle=False)

    # (B) 角度得分
    # 手动计算 b 值，方便核对
    # 论文公式: b = 0.002 * L
    b_val = 0.002 * dist_km
    e_angle = calc_angle_score(uav.pos, uav.velocity, target.pos)

    # (C) 速度得分
    uav_s_val = np.linalg.norm(uav.velocity)
    tgt_s_val = np.linalg.norm(target_vel)  # 这里只是为了计算score，mechanics里没用到target velocity vector
    e_speed = calc_speed_score(uav_s_val, tgt_s_val)

    # (D) 最终命中率
    p_hit = calc_damage_prob(uav, target)

    # (E) 经济账
    revenue = p_hit * target.value
    cost = uav.cost * cfg.COST_WEIGHT_OMEGA
    net_reward = revenue - cost

    # 3. 打印详细报告
    print(f"输入条件:")
    print(f"  - 距离: {dist_km} km")
    print(f"  - 夹角: {angle_deg} 度 (sigma)")
    print(f"  - UAV速度: {uav_speed_km_s * 1000:.0f} m/s")

    print(f"中间指标:")
    print(f"  - [距离评分 E_dist]:  {e_dist:.6f} (参数 zeta={cfg.PARAM_ZETA_D})")
    print(f"  - [角度评分 E_angle]: {e_angle:.6f} (参数 b={b_val:.4f})")
    print(f"  - [速度评分 E_speed]: {e_speed:.6f} (k={cfg.PARAM_K})")
    print(f"  - [载荷 Load]:        {uav.load:.2f}")

    print(f"计算结果:")
    print(f"  - 命中率 P_hit:       {p_hit:.6f} (= E_a * (0.75*E_d + 0.25*E_s) * Load)")
    print(f"  - 预期收益 Revenue:   {revenue:.6f} (= P_hit * 4.0)")
    print(f"  - 分配成本 Cost:      {cost:.6f} (= 1.0 * {cfg.COST_WEIGHT_OMEGA})")
    print(f"  - 净利润 J(X):        {net_reward:.6f}")

    if net_reward > 0:
        print(f"  >>> 判定: ✅ 盈利 (环境接受分配)")
    else:
        print(f"  >>> 判定: ❌ 亏本 (环境拒绝分配)")


def main():
    print("================ 参数检查 ================")
    print(f"当前 PARAM_ZETA_D (距离衰减): {cfg.PARAM_ZETA_D}")
    print(f"当前 COST_WEIGHT_OMEGA (成本权重): {cfg.COST_WEIGHT_OMEGA}")
    print("==========================================")

    # 场景 1: 初始状态 (距离远，角度一般)
    # 模拟环境刚Reset时的典型情况
    diagnose_scenario(
        name="初始状态 (远距离)",
        dist_km=140.0,
        angle_deg=10.0  # 假设运气好，只有10度偏差
    )

    # 场景 2: 中途状态 (距离适中)
    diagnose_scenario(
        name="中途状态 (中距离)",
        dist_km=80.0,
        angle_deg=5.0
    )

    # 场景 3: 理想攻击状态
    diagnose_scenario(
        name="贴脸输出 (近距离)",
        dist_km=20.0,
        angle_deg=2.0  # 近距离时角度非常敏感
    )


if __name__ == "__main__":
    main()