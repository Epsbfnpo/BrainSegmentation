# PPREMO/PREBO 400 Epoch 结果复盘

## 1. Per-class Dice 出现 NaN 的根因与解决方案
- **根因**：验证阶段的逐类 Dice 是直接用 `DiceMetric(reduction="none")` 聚合出的 raw 数组，没有携带 `not_nans` 掩码；一旦某个类别在一个 batch 中完全缺失（预测和标签都为 0），MONAI 会把该条目设为 NaN，这个 NaN 会被直接写进 `per_class_dice`。在当前的数据上，部分结构（例如日志里反复提示缺少采样体素的类别 24、67）在很多 patch 中完全缺席，于是最终的 per-class 统计不可避免地含有 NaN。【F:new/trainer_age_aware.py†L476-L605】
- **短期处置**：
  1. 在 `DiceMetric` 里打开 `get_not_nans=True`，拿到有效样本数，再在 `per_class_metric.aggregate()` 之后用该掩码把 NaN 剔除或替换为“无数据”。
  2. 生成报告时把 `torch.nan_to_num` 应用到 `per_class_scores`，同时给出“该类别在验证集中无 GT 体素”的提示，避免用户误解为数值爆炸。
- **长期方案**：
  1. 针对极度稀有的区域（如 class 24/67），考虑在数据加载阶段引入“至少包含这些 label 的 patch”采样策略；MONAI 的 `RandWeightedCropByLabelClassesd` 已经支持基于类别权重的 oversampling，可以把这些类别的权重调高。
  2. 如果医学上允许，也可以把在 PPREMO/PREBO 中极少出现的标签合并或在统计阶段与其左右对称区域共享一次 Dice 统计，避免出现完全缺失导致的 NaN。

## 2. 为什么除了 shape 以外的 prior loss 几乎不变？
- **量化观察**：metrics 曲线显示 volume / edge / spectral / rule / symmetry 这些项在 400 个 epoch 内几乎是一条平线，而 shape loss 会持续下降。
- **原因分析**：
  1. Volume、edge、required 等项都依赖 `flat = probs.view(C, -1)` 再做全局直方图或共现矩阵（`adj_pred = flat @ flat^T / N`）；如果模型一开始就加载了 dHCP 预训练权重，则这些全局统计和先验已经接近，梯度在前几轮后迅速饱和，自然维持在常数附近。【F:new/graph_prior_loss.py†L463-L588】
  2. 这些 loss 还要乘上 `lambda_factor = warmup * reliability`。对于 PPREMO/PREBO 里年龄段样本数量较少的 bin，`_age_reliability` 会落在 `age_reliability_min^pow`（≈0.3^0.5 ≈ 0.547）附近，从而把大部分先验项压得很低；在 warmup 没结束之前它们几乎没有贡献。【F:new/graph_prior_loss.py†L463-L522】
  3. Required / forbidden / symmetry 依赖的结构规则集合（`self.required_edges` 等）相对稀疏，当 `R_mask` 把绝大部分边置零以后，留给这些 loss 的有效元素不多，它们更容易进入“恒定”状态。
- **改进建议**：
  1. **提升有效梯度**：降低 `age_reliability_pow` 或适度提高 `age_reliability_min`，让可靠度最低的样本也能提供 ≥0.8 的权重；或者把 warmup 缩短到 5~8 epoch。
  2. **监控真实差异**：在 `analysis/structural_metrics.json` 中记录 `adj_mae`、`spec_gap` 等均值，若长期 <1e-3，可考虑重新标定先验，把 target 数据集自己的统计写入 priors，避免模型始终在匹配“另一个域”的结构。
  3. **增加触发条件**：对于 required/forbidden，可以把 `required_margin` 提高到 0.3、把 `forbidden_margin` 降到 1e-4，并在 `train_graphalign_age.py` 里设置一个“若 violations 连续 N 个 epoch 低于阈值则重新抽取难例”的回调，这样 loss 才会对结构错误敏感。
  4. **逐步恢复动态项**：`lambda_dyn` 目前只在 epoch ≥60 时生效（start=60，ramp=40），对于 400 epoch 的训练属于后 15% 才介入。可以把 `dyn_start_epoch` 调到 80×(新总 epoch/400) 后再观察，使得 dynamic scaling 真正覆盖训练后半段。

## 3. 把总 epoch 提升到 1000 需要做的准备
1. **调参顺序**：
   - 在 `run_training_graphalign.sh` 里把 `EPOCHS` 改成 1000，并同步放大与 epoch 相关的 scheduler：`LR_WARMUP_EPOCHS` 建议提升到 50~80，`LR_MIN` 也可以稍微减小（如 5e-8），保证 Cosine/LR decay 在 1000 epoch 内仍有足够动态区间。【F:new/run_training_graphalign.sh†L17-L94】
   - 根据新的总时长，把 `DYN_START_EPOCH` 与 `DYN_RAMP_EPOCHS` 等比例放大（例如 start≈150、ramp≈100），保持“动态先验只在模型稳定后才介入”的策略。【F:new/run_training_graphalign.sh†L105-L114】
2. **作业调度**：1000 epoch 意味着至少 2.5 倍的 wall-clock，需要确保自动续训逻辑有足够的检查点：
   - `train_graphalign_age.py` 已经在每次触发缓冲退出时写入 `latest_model.pt` 并更新 `resume_from.txt`，保持该机制不变即可。【F:new/train_graphalign_age.py†L682-L909】
   - 额外建议把 `checkpoint_keep_last` 策略设为“只保留每 50 epoch 一个 + 最佳模型”，否则 1000 个 epoch 会写入大量 `checkpoint_epoch*.pt`。
3. **监控与可视化**：长程训练更容易漂移，需要确保 `metrics_history.json` 和自动生成的曲线继续生效，同时在 `results/analysis` 里开启“每 50 epoch 导出结构先验的平均值”以快速发现某个 loss 长期不变的情况。【F:new/train_graphalign_age.py†L216-L1011】
