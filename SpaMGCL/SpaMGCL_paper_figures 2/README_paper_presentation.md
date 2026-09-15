# SpaMGCL 论文图表包

本包基于已有 notebook 输出和上传的训练节点记录绘图。模型、配置和原 notebook 未修改，未运行训练。

当前建议正文采用 2 张结果表和 3 类图。核心对比损失消融用图展示，完整数值表放补充材料，避免同一组数据在正文同时占用一张表和一张图。

## 正文放哪些图

| 内容 | 作用 | 当前状态 |
|---|---|---|
| 实际方法框架图 | 交代输入、空间/特征视图、对比目标和输出 | 待按最终方法定稿绘制 |
| 空间聚类结果图 | 展示标注、Core、去样本级对比的空间分区差异 | 待对应预测、spot IDs、坐标齐备后绘制 |
| 对比损失消融图 | 展示样本级与聚类级对比的贡献 | 已完成，fig_1_loss_ablation_ARI.pdf |

NMI 消融、200 轮读出比较、训练时长曲线放补充材料；正文讨论其中与结论有关的观察。图号可在排版时统一。

## 正文放哪些表

| 表 | 文件 | 范围 |
|---|---|---|
| 跨数据集结果表 | table_core_200.csv | 5 个数据集，Core，200 轮，Q 与 concat_z 两种读出 |
| SC/SNF 组件消融表 | table_spatial_50.csv | Core、SC、SNF、SC+SNF，50 轮，显式空间损失权重为 0 |

对比损失的完整数值表 table_ablation_50.csv 放补充材料，其中选 Core、no_sample、no_cluster 即对应正文消融图。

完整记录保存在 table_all_saved_results.csv。表格包含旧运行供追溯，不能将不同轮数或配置直接混为同条件对照。

## 图注与实验设置

可用于论文的图注和结果段落见 paper_captions.md。图面采用简短标签，不显示随机种子编号，不重复写实验次数、误差条或显著性说明。实验次数和随机性控制集中放在实验设置中一次说明。

Q output 即 Q_argmax；Embedding + k-means 即 concat_z 的 k-means 聚类。绘图标签调整不改变数据或算法。Core 仍包含空间图和特征图，SC/SNF 关闭不等于移除了所有空间信息。

## 本轮审查与修改

| 原呈现方式 | 修改方式 |
|---|---|
| 同一组核心损失消融在正文既放完整表又放图 | 正文保留图，完整数值表放补充材料 |
| 每张图反复写随机种子、单次运行和没有误差条 | 种子编号从图上移除；实验次数在设置中统一说明 |
| 图注重复写不能证明什么、还缺什么 | 图注只定义实验和读图方式；结果解释放正文讨论 |
| 使用 concat_z、长运行路径等实现标签 | 图面改为 Q output、Embedding + k-means；来源留在数据文件 |
| 把不同轮数的运行放在 200 轮标题下 | 按批次和轮数筛选，保留准确的 50/200 轮标注 |
| 用一句提升概括不同指标、读出和组件 | 分别报告 Q、表示聚类、ARI、NMI 及 SC/SNF 的实际变化 |

保留重复次数、训练轮数、Core 的定义和完整对照，是对实验协议的说明。本文稿没有均值、标准差、误差条或统计显著性主张，也没有将内部读出比较命名为外部方法比较。

## 数据与精度

- 来源 notebook：论文结果kaggle.ipynb，仅解析其保存的结果，不执行 cell。
- 消融和最终读出表来自 notebook 中四位小数的结果；绘图沿用这一精度。正式定稿时可使用对应原始 metrics.json 核对。
- 训练曲线使用 21 份上传的 metrics_epoch 文件的完整精度，均与 notebook 中的六位小数结果核对。
- NMI 统一沿用保存的指标字段。notebook 的 nmi_numpy 使用另一种归一化表达，未混入本次图表；论文评估设置需明确最终使用的归一化定义。
- figure_source.json 保留源文件哈希、表格 cell 编号、参数和结果来源；具体随机种子继续保留在该复现记录中。
- 已核验配置表列出的实验参数和开关；最终提交前应将原始配置与代码版本一并归档。

## 重绘

依赖 Python、NumPy、Matplotlib，无需 GPU、SpaMGCL 仓库或数据集。

```bash
python plot_paper_results.py --out redrawn
```

默认读取同目录 figure_source.json。只重绘图、不重写数据表时：

```bash
python plot_paper_results.py --figures-only --out redrawn
```

使用一份保存相同完整结果表的 notebook 更新来源：

```bash
python plot_paper_results.py --notebook /path/to/executed.ipynb --out redrawn
```

PDF 为矢量图，适合排版；PNG 为 300 dpi 预览图。

## 空间图所需文件

从对应的 50 轮 Core 和 no_sample 运行读取 pred_labels.npy、gt_labels.npy、spot_ids.npy、coords.npy。按 spot IDs 对齐，坐标范围、比例和点大小一致；颜色对应仅用于显示。用相同训练节点的预测对应消融数值。
