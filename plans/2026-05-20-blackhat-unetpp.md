# 形态学黑帽毛发掩膜辅助通道 — UNet++ 适配

**状态:** 完成
**创建:** 2026-05-20
**目标:** UNet++ 输入从 3 通道扩为 4 通道（RGB + 黑帽毛发掩膜），让网络自行学习抑制毛发。

## 已确认设计

| 决策 | 选项 |
|------|------|
| 预处理 | 另存 4 通道 .pt 文件 |
| 形态学库 | scikit-image |
| 核形状/尺寸 | disk(7) (圆形，直径~15px) |
| 文件夹 | `UNetPlusPlus-BlackHat/` |

## 文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `UNetPlusPlus-BlackHat/` | 新建 | 独立文件夹 |
| `preprocess_blackhat.py` | 新建 | 从现有 3 通道 .pt 生成 4 通道 .pt |
| `UNetPlusPlus-BlackHat/dataset.py` | 新建 | 加载 4 通道数据 |
| `UNetPlusPlus-BlackHat/model.py` | 新建 | NestedUNetV2(in_ch=4) |
| `UNetPlusPlus-BlackHat/config.py` | 复制 | 从 UNetPlusPlus 复制 |
| `UNetPlusPlus-BlackHat/train.py` | 复制 | 从 UNetPlusPlus 复制 |
| `UNetPlusPlus-BlackHat/eval.py` | 复制 | 从 UNetPlusPlus 复制 |

## 任务

### Task 1: 预处理脚本 `preprocess_blackhat.py` [x]
- 加载现有 `preprocessed/{split}_images.pt` (float16, [N,3,256,256])
- 逐张转灰度 → disk(7) 闭运算 → 黑帽 → 归一化到 [0,1]
- Concat 为 4 通道 → 保存到 `UNetPlusPlus-BlackHat/preprocessed/`
- 验证: 输出一张对比图（原图 / 黑帽掩膜 / 原图+掩膜叠加）

### Task 2: 模型 + 数据加载 [x]
- `dataset.py`: 加载 4 通道 .pt，normalize 只对前 3 通道用 ImageNet 统计量，第 4 通道不归一化
- `model.py`: NestedUNetV2 首层 `in_channels=3→4`
- 验证: 模型前向通过，输出形状正确

### Task 3: 验证训练 [x]
- 训练 2 epoch 不报错，loss 下降
