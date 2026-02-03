import numpy as np
import sys
import os

# 确保能导入项目模块
sys.path.append(os.getcwd())


def test_config_integrity():
    print("Test 1: Config Integrity (验证配置合理性)...", end=" ")
    from configs.config import cfg

    # 1. 验证关键维度
    assert cfg.UAV_STATE_DIM == 7, f"❌ UAV维度错误: 期望 7, 实际 {cfg.UAV_STATE_DIM}"
    assert cfg.TARGET_STATE_DIM == 4, f"❌ Target维度错误: 期望 4, 实际 {cfg.TARGET_STATE_DIM}"
    assert cfg.EDGE_DIM == 2, f"❌ Edge维度错误: 期望 2, 实际 {cfg.EDGE_DIM}"

    # 2. 验证兵种常量定义
    assert hasattr(cfg, 'TYPE_DECOY') and cfg.TYPE_DECOY == 0
    assert hasattr(cfg, 'TYPE_STRIKE') and cfg.TYPE_STRIKE == 1
    assert hasattr(cfg, 'TYPE_ASSESS') and cfg.TYPE_ASSESS == 2

    # 3. 验证场景生成接口
    scen = cfg.generate_scenario(num_uavs=10, num_targets=5)
    assert scen['n_uavs'] == 10
    assert scen['n_targets'] == 5
    assert 'type_ids' not in scen, "❌ 场景生成中仍包含旧的 'type_ids'，说明未清理干净"

    print("✅ Pass")


def test_entity_structure():
    print("Test 2: Entity Structure (验证实体数据结构)...", end=" ")
    from envs.entities import UAV, Target
    from configs.config import cfg

    # 1. 测试 UAV 初始化
    uav = UAV(id=0, pos=np.array([100, 100]))
    uav.reset(pos=np.array([0, 0]), v=np.array([1, 1]), u_type=cfg.TYPE_STRIKE)

    assert uav.uav_type == cfg.TYPE_STRIKE, "❌ UAV 类型设置失败"
    assert not hasattr(uav, 'fuel'), "❌ UAV 仍包含 'fuel' 属性，未清理干净"

    # 2. 测试 Target 初始化 (手动指定)
    tgt = Target(id=0, pos=np.array([500, 500]))
    tgt.reset(value=0.8, defense=5.0)

    assert tgt.value == 0.8, "❌ Target 价值设置错误"
    assert tgt.defense_level == 5.0, "❌ Target 防御等级设置错误"
    assert not hasattr(tgt, 'demands'), "❌ Target 仍包含 'demands' 旧属性"
    assert not hasattr(tgt, 'assigned_counts'), "❌ Target 仍包含 'assigned_counts' 旧属性"

    # 3. 测试 Target 初始化 (随机生成)
    tgt_rand = Target(id=1, pos=np.array([0, 0]))
    tgt_rand.reset()
    assert 0.0 <= tgt_rand.value <= 1.0, "❌ Target 随机价值越界"
    assert tgt_rand.defense_level >= 1.0, "❌ Target 随机防御等级异常"

    print("✅ Pass")


if __name__ == "__main__":
    print("================ 阶段一验证开始 ================")
    try:
        test_config_integrity()
        test_entity_structure()
        print("\n🎉 验证成功！第一阶段代码重构无误。")
        print("   - 配置已更新为功能异构模式")
        print("   - 实体类已精简并适配新物理属性")
    except AssertionError as e:
        print(f"\n🚫 验证失败: {e}")
    except Exception as e:
        print(f"\n🚫 发生未捕获异常: {e}")