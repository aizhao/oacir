# Q-Former Identity V2：一次组合增强实验

这是训练目标的组合实验，**不是已验证涨点，也不是新增推理架构**。
依赖服务器已安装的 `oacir_qformer_identity` 原模型；不修改原模型、原入口或 registry。
压缩包只增加一个模型文件、一个独立入口和本说明。

## 安装和训练

在 `/work/zhaoai/oacir` 解压，新文件分别进入 `lavis/models/blip2_models/` 和 `tools/`。
旧服务器的项目根目录是 `/home/caoyu/mnt/zhaoai/OACIR`；同样从该目录运行即可。
V2 自身兼容旧基础模型的缺失 bbox 接口，无需覆盖或修补原模型。
从项目目录执行：

```bash
python tools/run_oacir_qformer_identity_v2.py train
```

默认 Union、验证 Fashion/Car/Landmark、batch_size=256、lr=1e-5、seed=2026、50 epochs。
从 BLIP 预训练初始化，**不读取之前的 finetune checkpoint**。
自动检测 Union 原始视觉缓存和原有三个分类验证缓存。
只选择一张相对空闲的显卡，以完整 UUID 固定，并在加载模型前核对实际 UUID 和剩余显存。
显存容量读数差异只记录警告，不会再次错误阻止 UUID 已匹配的显卡。
不会结束任何进程。需要固定卡时使用 `--gpu 1`，数字会先解析为实际 UUID。

若要先少跑一些轮次，可显式加 `--num-epochs 20`。这会重新定义 OneCycleLR 的训练时长，
不是“50 轮训练的前 20 轮”，因此不能与旧 50 轮方案作严格同条件比较。

## 合并的改动

默认总损失：

`L = L_final + L_id_ignore + 1.0 * L_comp + 0.25 * L_comp_hard`

1. **组合 CE 权重从 0.5 提升为 1.0**。组合 CE 原本就是单向批内 InfoNCE，没有把 CE 换个名字当作新损失。
2. **身份 CE 排除同身份假负例**：仍只有对齐目标是正例；其他同身份目标从该损失分母排除。
   这不是多正例损失，不会把其他同身份图片作为正例主动拉近。
   身份标签来自真实标注完整父目录，保留分类命名空间，不按同名目录或图片相似度猜标签。
3. **组合路难负例 margin**：优先选 batch 内“同身份、不同目标图”中的最高组合分数；
   如果不存在，退回其他目标图中的最高分负例。重复目标图不作为负例。
   `L_comp_hard = mean(relu((hard_negative_cosine - target_cosine + 0.05) / 0.07))`。
   这里的 0.05 是原始余弦间隔，0.07 是固定缩放，不是可学习组合温度。
   同身份样本稀疏时仍有全局难负例监督，不额外更改 batch sampler。
4. **重复目标图保护**：最终检索 CE 和组合 CE 都排除 batch 中重复目标图的非对角条目，
   不把完全相同目标的另一次采样当作负例。不同图片即使同身份，在这两项准确目标监督中仍是负例。

身份特征分支仍不接收文本。最终检索损失仍反向传播到两路，**并非训练梯度完全解耦**。
新增 margin 复用同一前向的特征，只多做一个轻量 B×B 分数组合，不增加图像/文本编码器。
推理仍使用原公式和全图库两路打分；保持身份融合权重 1.25，以及原区域参数和文本长度 32。
所有学习参数名称/形状与原模型相同，但 checkpoint 的 model_name 独立，防止旧入口误加载。

## 时间和选模

不再每个改动单独训练 50 轮，只跑这一组组合实验。
默认前期每 5 轮验证一次，最后 10 轮逐轮验证：50 轮时验证 epoch
`0,5,10,15,20,25,30,35,40,41,42,43,44,45,46,47,48,49`。
全图库验证次数从 50 次减为 18 次，**不是承诺总耗时减少相同比例**；也可能错过未验证轮次的峰值。
如需恢复逐轮验证，用 `--validation-frequency 1`。

输出默认位于 `checkpoints/qformer_identity_v2_from_pretrain/train_<timestamp>/`，不覆盖旧实验。
保存 `best_identity.pt`、`best_image.pt`、`best_primary3.pt`，以及记录各自 epoch 的 `best_metrics.json`。
训练 CSV 额外记录 `same_id_hard_fraction`、身份忽略条目数和重复图条目数，便于确认新监督真正生效。
结果继续同时报告 R_ID@1 与 R@1；不以单类各自最好的轮次拼接结果。

## 验证新 checkpoint

```bash
python tools/run_oacir_qformer_identity_v2.py evaluate --checkpoint /实际路径/best_identity.pt
```

仅验证 Product：在上述命令中加 `--dataset Product`。
旧权重搜索/消融入口限定原 model_name，不支持直接读 V2 checkpoint；使用本 V2 入口验证。

## 验证边界

本地只能做静态代码与 NumPy/CPU 模拟逻辑检查，没有 PyTorch 运行环境。
未进行真实 PyTorch 前向/反向、预训练加载、服务器 GPU、真实数据或训练效果验证。
所有组合改动仍需实际实验判断；本轮不能归因到某一个改动，也不保证提升。
