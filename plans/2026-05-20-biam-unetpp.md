# BIAM 模块缝合到 UNet++

**状态:** 完成
**创建:** 2026-05-20
**目标:** 在 NestedUNetV2 解码器中嵌入 FDDBA-NET 的 BIAM 模块(Eq 6-12)，构建 BIAM-UNet++ 新模型。

## 方法概述

严格按原论文 Eq 6-12 实现三层双向交互注意力。论文中 BIAM 用于 U 型密集网络（即 UNet++），节点索引 (i, j) 与 NestedUNetV2 完全对应。每个解码节点 X[i][j] (j≥1) 接收三层特征：
- **浅层**: X[i-1][j]（i≥1 时存在）
- **同层**: X[i][j-1]
- **深层**: X[i+1][j-1]（i+1 < depth 时存在）

## 文件（新建独立文件夹，不动原仓库）

| 文件 | 操作 | 说明 |
|------|------|------|
| `UNetPlusPlus-BIAM/` | 新建文件夹 | 独立于原 UNetPlusPlus |
| `UNetPlusPlus-BIAM/biam.py` | 新建 | BIAM 核心模块（Transformer、注意力门、三层融合）|
| `UNetPlusPlus-BIAM/model.py` | 新建 | 完整 BIAM-UNet++，从 NestedUNetV2 复制 + 修改解码器 |
| `UNetPlusPlus-BIAM/dataset.py` | 复制 | 从 UNetPlusPlus 复制 |
| `UNetPlusPlus-BIAM/config.py` | 复制 | 从 UNetPlusPlus 复制 |
| `UNetPlusPlus-BIAM/train.py` | 复制 | 从 UNetPlusPlus 复制（输出接口不变）|
| `UNetPlusPlus-BIAM/eval.py` | 复制 | 从 UNetPlusPlus 复制 |

## 公式到代码映射

### 空间分辨率约定

NestedUNetV2 depth=5 (level 0-4)，分辨率依次: 256→128→64→32→16。
节点 X[i][j] 的空间分辨率 = level i。

### 节点计算流程（替代原 concat+conv）

```
原 (NestedUNetV2):
  up = upsample(X[i+1][j-1])
  cat = concat(up, X[i][0], X[i][1], ..., X[i][j-1])
  X[i][j] = DoubleConv(cat)

新 (BIAM):
  Step 1: Skip 层融合 (Eq 5) — 远处 skip 直接拼
    X_skip = concat(X[i][j-2], X[i][j-3], ..., X[i][0])  # 不含 X[i][j-1]
    
  Step 2: 相邻层融合 (Eq 6-9) — 三层 BIAM 调制
  
    2a. 下调制 Eq 6: 浅层→深层
      same_proj = BN(conv1(X[i][j-1]))
      attn = sigmoid(TransformerEncoder2D(same_proj))
      shallow_aligned = MaxPool(X[i-1][j])  # level i-1 → level i
      X_down = attn ⊗ shallow_aligned
    
    2b. 上调制 Eq 7: 深层→浅层
      same_gate = sigmoid(BN(conv3(ReLU(BN(conv2(X[i][j-1]))))))
      deep_aligned = Upsample(X[i+1][j-1])  # level i+1 → level i
      X_up = same_gate ⊗ deep_aligned
    
    2c. 同层调制 Eq 8: 深+浅双向调制同层
      deep_aligned = Upsample(X[i+1][j-1])  # level i+1 → level i
      attn_deep = sigmoid(TransformerEncoder2D(BN(conv4(deep_aligned))))
      same_pooled = MaxPool(X[i][j-1])  # level i → level i+1
      attn_same_raw = sigmoid(BN(conv6(ReLU(BN(conv5(same_pooled))))))
      attn_same = Upsample(attn_same_raw)  # level i+1 → level i (论文隐式)
      X_same = attn_deep ⊗ X[i][j-1] ⊗ attn_same
    
    X_bd = concat(X_down, X_same, X_up)  # Eq 9
  
  Step 3: 最终融合 (Eq 12)
    combined = concat(X_skip, X_bd) if X_skip 非空 else X_bd
    X[i][j] = DoubleConv(combined) + residual(combined)  # R = 残差连接
```

### 边界情况处理

**i = 0（无浅层，Eq 11）:** 
跳过 X_down，X_bd = concat(X_same, X_up)

**i = depth-2（无深层）:**
跳过 X_up（以及 X_same 中依赖 deep 的项）。由于最深解码行只有 j=1，只剩 X[3][0] 和 X[2][1]。
若仍需求三层，X_same 中 deep 相关的项置为单位映射。

**j = 1（无 X_skip）:**
X_skip 为空，只使用 X_bd

### TransformerEncoder2D 设计

```
输入: [B, C, H, W]
  1. Patch Embedding: Conv2d(C, C, kernel=1)  # 逐点投影
  2. 加入可学习位置编码 [1, C, H, W]
  3. Flatten: [B, C, H*W] → transpose → [B, H*W, C]
  4. TransformerEncoderLayer (d_model=C, nhead=8, mlp_ratio=4)
  5. Reshape: [B, H*W, C] → [B, C, H, W]
输出: [B, C, H, W]
```

## 任务

### Task 1: 创建目录 + 复制共享文件 [x]

- **文件**: `UNetPlusPlus-BIAM/`
- **验证**: 文件夹存在，config.py / dataset.py / train.py / eval.py 内容与 UNetPlusPlus 一致
- **内容**: 从 UNetPlusPlus 复制 dataset.py, config.py, train.py, eval.py

### Task 2: 实现 BIAM 核心模块 (`biam.py`) [x]

- **文件**: `UNetPlusPlus-BIAM/biam.py`
- **验证**: `cd UNetPlusPlus-BIAM && python -c "from biam import TransformerEncoder2D, BIAMLayer; ..."` 各子模块前向通过
- **内容**:
  1. `TransformerEncoder2D`: Patch embedding + position encoding + MHA + FFN → 2D 输出
  2. `ChannelGate`: 1×1Conv → BN → ReLU → 1×1Conv → BN → Sigmoid
  3. `BIAMLayer(ch_in, ch_same, ch_out)`: 单层三层融合，包含 Eq 6/7/8 三条路径
  4. `BIAM(depth, filters)`: 管理 depth-1 层 BIAMLayer 的容器

### Task 3: 编写 BIAM-UNet++ 模型 (`model.py`) [x]

- **文件**: `UNetPlusPlus-BIAM/model.py`
- **验证**: `m = NestedUNetV2(3,1); outs = m(torch.randn(2,3,256,256))` 输出 4 个头，参数 <20M
- **内容**:
  1. 从原 `NestedUNetV2` 复制编码器 + 深度监督头结构
  2. 解码器部分：每个 X[i][j] (j≥1) 用 BIAMLayer + DoubleConv 替代原 concat+DoubleConv
  3. 边界处理：i=0 走 2 路（Eq 11），最深行走 2 路

### Task 4: 验证训练流程 [x]

- **文件**: `UNetPlusPlus-BIAM/train.py`
- **验证**: `python train.py` 训练 2 epoch 不报错，loss 下降
- **内容**: 确认现有 train.py 可直接加载新模型（输出 [L1,L2,L3,L4] 不变）

