# 快速开始

GemFitCom 由四个 CLI 子命令组成，每个对应 pipeline 的一个阶段。
阶段之间通过磁盘上的 artifact（YAML、CSV、SBML）通信，所以每一步
都可审计、可重放。

仓库自带一份 `data/examples/` 下的合成 mini 数据集，校准这一步
不依赖任何外部下载就能端到端跑起来。需要重新生成的话：
`python scripts/build_example_data.py`。

## 1. 校准单菌株动力学

```bash
gemfitcom fit configs/example_strain.yaml --output results/
```

这一步加载 OD 曲线，可选地跑一次 KB 驱动的 gap-fill，然后为配置好的
碳源拟合 V<sub>max</sub> 和 K<sub>m</sub>。在示例数据集上
（真值：V<sub>max</sub> = 4.0，K<sub>m</sub> = 2.0）大致输出：

```
fit: strain=toy_acetogen R²=1.0000 Vmax=3.877 Km=1.831
  params → results/toy_acetogen_fitted_params.yaml
  model  → results/toy_acetogen_model_fitted.xml
  grid   → results/toy_acetogen_fit_grid.csv
```

`params` 文件给第 2 步用；`model` 文件是带交换通量上下界约束的 SBML；
`grid` 文件是 V<sub>max</sub> × K<sub>m</sub> 的 R² 表，
`viz.plot_kinetics_heatmap` 会用它作图。

## 2. 模拟群落

```bash
gemfitcom simulate configs/example_community.yaml --output results/
```

通过配置里的 `simulation.mode` 选三种模式之一：

- `sequential_dfba` —— 每个时间步对每个菌株独立做 FBA，共享代谢物池。
  快，适合看轨迹细节。
- `micom` —— MICOM 稳态合作权衡群落 FBA。没有时间维度，是一个快照。
- `fusion` —— 动态 dMICOM（每个 dFBA 时间步跑一次合作 MICOM 优化）。
  最慢但最有表达力。

输出包括 `exchange_panel.csv` 和 `biomass_panel.csv`；dFBA 模式还会额外
保存完整的 `biomass.csv` 和 `pool.csv` 轨迹。

## 3. 推导交互边和网络

```bash
gemfitcom interactions \
    results/demo_community_exchange_panel.csv \
    --biomass results/demo_community_biomass_panel.csv \
    --output results/network/
```

交叉饲喂边（donor → recipient）写到一个 CSV；竞争边（成对的共同摄取）
写到另一个 CSV；以及一个合并的 GraphML 文件供下游可视化用。
用 `viz.plot_interaction_network` 在 Python 里渲染 GraphML。

## 4. 单独跑 gap-fill

```bash
gemfitcom gapfill path/to/model.xml \
    --source agora2 --observed EX_ac_e,EX_but_e \
    --output results/
```

适用场景：只想给 GEM 加缺失的发酵产物交换反应，不需要跑完整校准。
KB 驱动的方法只添加最少的反应集，让每个观察到的产物可被生产。

## 帮助和选项列表

`gemfitcom COMMAND --help` 显示任意子命令的所有 flag。完整的配置
schema 见 `configs/example_strain.yaml` 和 `configs/example_community.yaml`。
