# UGTST 如何在 3D MRI 上复用 2D 模型

为了承接 PPREMO/PREBO 数据集的 3D MRI 与 UGTST 原本基于 2D 切片的训练/推理范式，我在 `code_oa/dataloaders/ppremo_dataset.py` 中构建了一个双轨数据管线：它对 3D 体数据执行与 `new` 方法相同的几何、强度与标签处理，然后在训练阶段自动切片给 2D 网络使用，在验证阶段则保留整块体数据用于滑动评估。核心思路分为以下三步：

## 1. 3D 层面的预处理保持一致

`DatasetConfig` 和 `_base_transforms` 定义了统一的 3D 预处理策略：

- `Spacingd` / `Orientationd` 将所有 volume 重采样到同一体素间距与 RAS 朝向，确保网络看到的物理几何与 `new` 文件夹完全一致。
- `PercentileNormalizationd` 在脑掩模内做 1-99 分位数强度归一化，对齐强度分布。
- `RemapLabelsd` 把 Draw-EM 风格的 88 类标签映射成 87 个前景 + 忽略标签（-1），从而复用 `new` 方法的 foreground-only 设定。
- `SpatialPadd` + `CenterSpatialCropd` 把体数据裁剪/填充到 128×128×128 的固定 ROI，确保 2D 切片来自同一个 3D 视野。

这些步骤都直接作用在 3D tensor 上，所以即使最终喂入 2D 网络，也能保证输入已经按 3D 规则完成对齐。`train_finetune.py` 中通过 `create_target_datasets` 显式传入 `--roi_x/y/z` 与 `--target_spacing` 等参数，从命令行控制这条 3D 预处理链。 【F:UGTST-main/code_oa/dataloaders/ppremo_dataset.py†L104-L191】【F:UGTST-main/code_oa/train_finetune.py†L23-L104】

## 2. 训练阶段：3D 切片化为 2D 批次

`PPREMOSliceDataset` 在初始化时先把所有 case 经过上述 3D 预处理，然后构建 `(case_idx, slice_idx)` 的映射表。`__getitem__` 会在深度维（Z 轴）上选取一张轴向切片，返回的 `image` 张量 shape 为 `(C, X, Y)`，`label` 为 `(X, Y)`，与原始 UGTST 2D pipeline 的输入保持完全一致。这样既能利用 3D 预处理成果，又能继续沿用 UGTST 的 Two-Stream Batch Sampler 和半监督损失设计。 【F:UGTST-main/code_oa/dataloaders/ppremo_dataset.py†L223-L283】

## 3. 验证/测试：整块 3D 输入再切片评估

`PPREMOVolumeDataset` 会输出完整的 128³ patch，并在 `val_2D.py` 中逐切片送入 2D 模型，再把各切片的 softmax 重新拼回 3D volume，从而计算 87 个结构的 Dice。这样做既维持了 2D 模型推理逻辑，也保证最终指标是在 3D 层面评估的，与 `new` 方法完全对齐。训练脚本 `train_finetune.py` 中的 `valloader` 就读取 `PPREMOVolumeDataset`，确保验证集不再丢失任意维度信息。 【F:UGTST-main/code_oa/dataloaders/ppremo_dataset.py†L193-L221】【F:UGTST-main/code_oa/train_finetune.py†L105-L182】【F:UGTST-main/code_oa/val_2D.py†L1-L200】

通过上述三步，UGTST 的 2D 模型就可以在不改动网络结构的前提下，严格遵循 3D 数据的几何/强度/标签设定，实现与 `new` 管线公平、可复现的对比实验。
