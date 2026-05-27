# 训练启动命令

## 第一步：预处理数据（只需运行一次）

将 parquet 中的 JPEG 图像预先 resize 为 256×256 tensor 并存入磁盘，后续训练不再重复解码。

```powershell
python preprocess.py
```

输出：`preprocessed/` 目录 (train/val/test 的 images 和 labels，共 6 个 .pt 文件)

### CLAHE 增强预处理

为 CBAM-CLAHE 和 CBAM-ASPP-CLAHE 生成对比度增强数据：

```powershell
cd UNetPlusPlus-CLAHE
python preprocess.py

cd ../UNetPlusPlus-CBAM-CLAHE
python preprocess.py

cd ../UNetPlusPlus-CBAM-ASPP-CLAHE
python preprocess.py
```

输出：各自 `preprocessed/` 目录 (CLAHE 增强 .pt 文件)

## 第二步：训练

```powershell
# Attention U-Net (31.40M params)
cd AttentionUNet
python train.py

# ResUNet (32.44M params)
cd ../ResUNet
python train.py

# U-Net++ (9.16M params, 深度监督)
cd ../UNetPlusPlus
python train.py


# UNet++ CBAM (9.22M params, 编码器+解码器嵌入 CBAM 注意力)
cd ../UNetPlusPlus-CBAM
python train.py

# UNet++ CBAM + ASPP (11.98M params, 瓶颈 ASPP 多尺度融合)
cd ../UNetPlusPlus-CBAM-ASPP
python train.py

# UNet++ + CLAHE (9.16M params, 消融：纯 UNet++ + CLAHE 预处理)
cd ../UNetPlusPlus-CLAHE
python train.py

# UNet++ CBAM + CLAHE (9.22M params, LAB L-channel CLAHE 预处理)
cd ../UNetPlusPlus-CBAM-CLAHE
python train.py

# UNet++ CBAM + ASPP + CLAHE (11.98M params, 瓶颈 ASPP + CLAHE 预处理, epoch=50)
cd ../UNetPlusPlus-CBAM-ASPP-CLAHE
python train.py
```

## 第三步：评估（训练完成后）

```powershell
cd AttentionUNet
python eval.py

cd ../ResUNet
python eval.py

cd ../UNetPlusPlus
python eval.py


cd ../UNetPlusPlus-CBAM
python eval.py

cd ../UNetPlusPlus-CBAM-ASPP
python eval.py

cd ../UNetPlusPlus-CLAHE
python eval.py

cd ../UNetPlusPlus-CBAM-CLAHE
python eval.py

cd ../UNetPlusPlus-CBAM-ASPP-CLAHE
python eval.py
```

## 输出

- 最佳模型权重保存在 `checkpoints/best.pth`
- 可视化结果保存在 `visualizations/`
- 训练日志直接输出到终端
