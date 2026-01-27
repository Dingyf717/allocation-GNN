# envs/mechanics.py
import numpy as np


def get_distance(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)


def calc_angle_score(uav_vel, uav_pos, target_pos):
    """
    计算飞行角度优势：越指向目标，分数越高 (0~1)
    """
    vec_u_t = target_pos - uav_pos
    dist = np.linalg.norm(vec_u_t)
    if dist < 1e-3: return 1.0

    # 归一化
    vec_u_t /= dist
    speed = np.linalg.norm(uav_vel)
    if speed < 1e-3: return 0.5  # 静止时给个中间分
    vec_v = uav_vel / speed

    # Cosine Similarity: [-1, 1] -> 映射到 [0, 1] (可选)
    # 或者直接用 cos 值作为奖励的一部分
    cos_theta = np.dot(vec_u_t, vec_v)
    return cos_theta  # Range: -1.0 to 1.0 (背对惩罚，正对奖励)


def calc_reward(uav, target, action):
    """
    计算单步奖励
    """
    # 1. 基础物理惩罚 (距离越远成本越高)
    dist = get_distance(uav.pos, target.pos)
    dist_penalty = -0.001 * dist  # 系数根据地图大小调整

    # 2. 角度奖励 (鼓励顺路分配)
    angle_score = calc_angle_score(uav.velocity, uav.pos, target.pos)
    angle_reward = 0.5 * angle_score

    if action == 0:  # Skip
        # 只有当该无人机确实没有合适去处时，Skip才不惩罚
        # 这里简单处理：给一个微小的存在惩罚，鼓励干活
        return -0.1

    # Action == 1 (Assign)
    # 3. 供需匹配逻辑
    needed, assigned = target.get_demand_status(uav.uav_type)

    if needed > 0:
        # 【成功分配】
        # 奖励 = 基础分 + 角度加成 - 距离成本
        return 10.0 + angle_reward + dist_penalty
    else:
        # 【无效分配】(不需要或已满)
        # 给予较大惩罚，教会它不要乱选
        return -5.0