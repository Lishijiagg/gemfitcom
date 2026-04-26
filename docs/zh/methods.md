# 方法学

## 动力学拟合（V<sub>max</sub>、K<sub>m</sub>）

GemFitCom 通过最小化 dFBA 模拟得到的生物量轨迹与 OD 推导得到的
生物量曲线之间的 $1 - R^2$，来估计 Michaelis–Menten 参数。
R² 是在 **原始**（raw）生物量尺度上计算的，**不归一化** —— 因为
把两条轨迹各自除以它们的最大值，会丢掉绝对尺度信息，并制造一个
结构性可识别脊（identifiability ridge），在这个脊上差别很大的
`(vmax, km)` 组合都能拿到接近 1.0 的分数。

最小化分两个阶段：

1. **全局搜索**：用 `scipy.optimize.differential_evolution` 在用户指定的
   `(vmax_bounds, km_bounds)` 矩形里搜索。默认参数比较保守
   （`maxiter=50`、`popsize=15`），让 wall time 可控。
2. **局部精修**：在以 DE 最优为中心的 `grid_points × grid_points` 均匀网格上
   做精修，每个维度的网格跨度是 `±grid_span × optimum`。这个网格一身二任：
   既精修 DE 最优，又顺便提供参数敏感度热图所需的 R² 表面。

### 可识别性的前提

由于在高底物浓度下摄取速率近似为 V<sub>max</sub>，生物量曲线只有在
观察时间窗内底物**真的被消耗掉了**，才能同时约束 V<sub>max</sub> 和
K<sub>m</sub>。如果 5 mM 葡萄糖在 6 小时内几乎没被消耗，你能恢复出
V<sub>max</sub>，但拿不准 K<sub>m</sub>。设计实验时要保证时间足够长，
看到减速期。

## 三种群落模拟模式

`simulate` 子命令暴露了三种积分方案：

### `sequential_dfba`

经典动态 FBA 循环：每个时间步，每个菌株独立解自己的 FBA，
当前共享池浓度作为摄取的上界。池更新按每菌株生物量加权，汇总每菌株通量。
便宜，能暴露时间分辨的交叉饲喂。

### `micom`

一次 MICOM 优化：群落 FBA 找一组通量分配，满足"个体最大生长"和
"群落总生长"之间的合作权衡，由 `tradeoff_alpha ∈ [0, 1]` 参数化。
α 较小偏向严格的群落最优，α 较大趋近每个菌株的无约束最大值。
没有时间维度 —— 结果是一个快照。

### `fusion`（dMICOM）

GemFitCom 的贡献：在每个 dFBA 时间步上，不再对每个菌株独立做 FBA，
而是对整个群落跑一次 MICOM 合作权衡优化。这产生了在整个模拟过程中
都尊重合作性的动态轨迹 —— 当交叉饲喂随时间演化、合作效应即使在
曲线早期也很重要时，特别有用。

## Gap-fill 策略

对于标记为 `model_source: curated` 的模型，gap-fill 是空操作。
对于 `agora2` 或 `carveme` 模型，KB 驱动的 gap-fill 把观察到的 HPLC 产物
和模型当前的交换列表做对比：

- 对每个观察到但模型当前不能产生的产物，KB 用一组 `(reactions, metabolites)`
  描述其规范的生物合成途径，用来添加。
- 反应和代谢物只在尚未存在时才添加。
- 一份报告记录了添加了什么、什么本来就有、KB 中没有对应条目的是哪些。

默认的知识库自带七条 SCFA 途径（acetate、butyrate、propionate、formate、
lactate-D、lactate-L、succinate），位于
`src/gemfitcom/data/gapfill_kb/scfa.yaml`。自定义 KB 遵循同一个 schema，
通过 `--kb` 传入。
