# BIAM-UNet++ 低分辨率层辅助深监督

**状态:** 完成
**创建:** 2026-05-20
**目标:** 在 row 1 (128×128) 和 row 2 (64×64) 添加辅助分割头，让低分辨率层 BIAM 模组直接受监督，避免退化。

## 方法概述

当前深监督仅在 row 0（全分辨率），BIAM 所在的 row 1/2/3 梯度间接回传太弱。
在 X[1][1], X[1][2], X[2][1] 加辅助头，上采样回 256×256 算 loss。

## 文件

| 文件 | 操作 |
|------|------|
| `UNetPlusPlus-BIAM/model.py` | 添加 aux_heads 辅助输出头 |
| `UNetPlusPlus-BIAM/config.py` | 添加 STAGE2 参数 (CKPT, AUX_WEIGHTS, LR 倍率) |
| `UNetPlusPlus-BIAM/train.py` | 辅助 loss + 加载 stage1 ckpt + stage2 训练逻辑 |

## 任务

### Task 1: model.py 添加辅助输出头 [x]
- 在 X[1][1], X[1][2], X[2][1] 加 1×1 Conv 输出单通道
- forward 返回 (main_outputs, aux_outputs)
- 验证: 输入 (2,3,256,256) → aux 输出形状正确

### Task 2: config.py 添加 stage2 参数 [x]
- STAGE1_CKPT, STAGE2_LR_FACTOR=0.1, STAGE2_EPOCHS=50, AUX_WEIGHTS
- 验证: config 导入检查

### Task 3: train.py 辅助损失 + 两阶段训练 [x]
- compute_loss() 接受 aux outputs
- 加载 stage1 checkpoint (strict=False)
- LR * 0.1, 训练 50 epoch
- 验证: 训练 1 epoch 不报错
