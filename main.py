# main.py
import numpy as np
import time
from configs.config import cfg
from envs.uav_env import UAVEnv


def print_separator(title):
    print(f"\n{'=' * 30} {title} {'=' * 30}")


def run_diagnostic_test():
    print_separator("环境逻辑诊断测试 (Diagnostic Test)")

    # 1. 强制设置一个小规模场景方便观察
    # 我们临时覆盖 generate_scenario 的默认参数，或者依赖随机生成但打印出来
    print(f"Loading Config... Map Size: {cfg.MAP_WIDTH}x{cfg.MAP_HEIGHT}")
    print(f"State Dim: {cfg.STATE_DIM} (Should be 7)")

    env = UAVEnv()

    # 重置环境 (生成新场景)
    state = env.reset(full_reset=True)

    # --- 打印场景概览 ---
    print_separator("1. 场景生成概览")
    print(f"UAVs Count:    {len(env.uavs)}")
    print(f"Targets Count: {len(env.targets)}")

    # 统计总需求
    total_demand = {}
    for t in env.targets:
        for type_id, count in t.demands.items():
            total_demand[type_id] = total_demand.get(type_id, 0) + count

    # 统计 UAV 类型分布
    uav_types = {}
    for u in env.uavs:
        uav_types[u.uav_type] = uav_types.get(u.uav_type, 0) + 1

    print(f"UAV Supply:    {dict(sorted(uav_types.items()))}")
    print(f"Target Demand: {dict(sorted(total_demand.items()))}")
    print("------------------------------------------------------------")
    if sum(total_demand.values()) > len(env.uavs):
        print("⚠️ 注意: 总需求 > 总供给，不可能达到 100% 满足率。")
    else:
        print("✅ 供给充足 (或大致平衡)。")

    # --- 运行单回合测试 ---
    print_separator("2. 逐步逻辑验证 (Greedy Policy)")
    print(
        f"{'Step':<5} | {'UAV_ID':<6} {'Type':<4} -> {'Tgt_ID':<6} | {'Dist':<6} {'Angle':<6} | {'Need':<4} {'Sat?':<4} | {'Action':<8} | {'Reward':<8} | {'Check'}")
    print("-" * 100)

    done = False
    step_cnt = 0
    total_reward = 0
    valid_assigns = 0

    while not done:
        # 获取当前指针对象 (注意：env.step调用后指针会移位，所以要在step前获取)
        # 边界保护：防止 done=True 后索引越界
        if env.uav_idx >= len(env.uavs): break

        curr_uav = env.uavs[env.uav_idx]
        curr_tgt = env.targets[env.target_idx]

        # --- 策略逻辑: 有需求就分配 (Greedy) ---
        # 我们可以直接读 State，也可以直接查 Target 对象
        # 状态向量: [dist, angle, needed_norm, assigned_norm, is_satisfied, available, res]
        current_state = state[-1]  # 取序列最后一步

        feat_needed_norm = current_state[2]
        feat_is_satisfied = current_state[4]

        # 反归一化 (假设 env 中除以了 10.0)
        est_needed = int(round(feat_needed_norm * 10.0))

        # 真实值检查
        real_needed, _ = curr_tgt.get_demand_status(curr_uav.uav_type)

        # 动作选择
        if real_needed > 0:
            action = 1  # Assign
            act_str = "ASSIGN"
        else:
            action = 0  # Skip
            act_str = "Skip"

        # 记录执行前的数据用于对比
        prev_demand = curr_tgt.demands.get(curr_uav.uav_type, 0)

        # --- 执行 ---
        next_state, reward, done, info = env.step(action)

        # --- 验证逻辑 ---
        check_msg = "✅"

        # 1. 验证 State 是否准确
        if est_needed != real_needed:
            check_msg = f"❌ State Error (Obs:{est_needed} vs Real:{real_needed})"

        # 2. 验证 Reward 和 状态更新
        if action == 1:
            if real_needed > 0:
                # 期望：正奖励，且需求 -1
                if reward < 0:
                    check_msg = f"❌ Reward Error (Should be >0, got {reward:.2f})"
                elif curr_tgt.demands[curr_uav.uav_type] != prev_demand - 1:
                    check_msg = f"❌ Logic Error (Demand not decreased)"
                else:
                    valid_assigns += 1
            else:
                # 期望：负奖励 (惩罚乱分配)
                if reward > 0: check_msg = "❌ Reward Error (Should be <0)"

        # 格式化打印 (只打印前 20 步和关键步，避免刷屏)
        if step_cnt < 15 or action == 1:
            dist_val = get_dist(curr_uav.pos, curr_tgt.pos)
            # 角度分我们没法直接拿，只能从 reward 反推或者不管

            print(f"{step_cnt:<5} | {curr_uav.id:<6} {curr_uav.uav_type:<4} -> {curr_tgt.id:<6} | "
                  f"{dist_val:<6.1f} {'--':<6} | {real_needed:<4} {int(feat_is_satisfied):<4} | "
                  f"{act_str:<8} | {reward:<8.2f} | {check_msg}")

        state = next_state
        total_reward += reward
        step_cnt += 1

        # 防止死循环 (虽然 env 有 done 机制)
        if step_cnt > 2000:
            print("Force Break!")
            break

    print_separator("3. 最终结果验证")
    print(f"Total Steps:  {step_cnt}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Valid Assigns:{valid_assigns}")

    # 统计最终满足率
    total_assigned = 0
    total_needed_initial = sum(total_demand.values())

    print("\n[各目标满足情况]:")
    for t in env.targets:
        orig = sum(t.assigned_counts.values()) + sum(t.demands.values())
        curr = sum(t.assigned_counts.values())
        print(f"  - Target {t.id}: {curr}/{orig} (Unfilled: {t.demands})")
        total_assigned += curr

    sat_rate = total_assigned / total_needed_initial if total_needed_initial > 0 else 0
    print(f"\n>> 全局满足率 (Satisfaction Rate): {sat_rate * 100:.1f}%")

    if sat_rate > 0.5:
        print("\n🎉 测试通过！环境逻辑看似正常。智能体若能学会，应该能达到更高满足率。")
    else:
        print("\n⚠️ 测试警告：满足率较低。如果是 Greedy 策略，这可能意味着供给不足或逻辑有漏洞。")


def get_dist(p1, p2):
    return np.linalg.norm(p1 - p2)


if __name__ == "__main__":
    run_diagnostic_test()