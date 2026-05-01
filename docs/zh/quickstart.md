# 快速开始

GemFitCom 由四个 CLI 子命令组成，每个对应 pipeline 的一个阶段。
阶段之间通过磁盘上的 artifact（YAML、CSV、SBML）通信，所以每一步
都可审计、可重放。

仓库自带一份 `data/examples/` 下的合成 mini 数据集，校准这一步
不依赖任何外部下载就能端到端跑起来。需要重新生成的话：
`python scripts/build_example_data.py`。

## 输入数据格式

实验数据以 **CSV/TSV 文件**（长格式）提供。YAML 配置文件
（`configs/example_strain.yaml`）**不是**数据本身 —— 它指向 CSV
文件、设置每个菌株的参数。结构请参考 `data/examples/` 下的可运行
mini 数据集。

### OD 生长曲线（`*_od.csv`）

每行一个 `(time_h, carbon_source, replicate)` 组合：

| 列名            | 类型  | 含义                                              |
| --------------- | ----- | ------------------------------------------------- |
| `time_h`        | float | 接种后的小时数；t=0 是初始点                       |
| `carbon_source` | str   | 碳源条件标签（如 `glc__D`）                        |
| `replicate`     | int   | 1, 2, 3, …                                        |
| `od`            | float | OD600 读数                                         |

需要在 strain config 里声明 OD 是否已减除 baseline，以及给出
`initial_biomass_gDW_per_L`，让拟合器能够还原绝对生物量量级。

### HPLC 代谢物面板（`*_hplc.csv`）

每行一个 `(time_h, carbon_source, metabolite, replicate)` 组合：

| 列名            | 类型  | 含义                                              |
| --------------- | ----- | ------------------------------------------------- |
| `time_h`        | float | 接种后的小时数；**可选** —— 端点测量数据可留空    |
| `carbon_source` | str   | 与 OD 文件里的 `carbon_source` 列对齐              |
| `metabolite`    | str   | 显示名（`acetate`, `butyrate`, `propionate`, `lactate`, …） |
| `value_mM`      | float | 浓度，单位毫摩尔                                   |
| `replicate`     | int   | 1, 2, 3, …（缺省 1）                               |

真实的 HPLC 实验通常采集多个时间点、每次进样输出多种代谢物，
所以 canonical 形式是时间序列。仅有终点测量的表格（没有 `time_h`）
依然能加载 —— 内部会按单一 endpoint snapshot 处理。

### 文件格式备注

- 接受 TSV：如果自动检测识别不出分隔符，给 `load_od` / `load_hplc`
  传 `sep="\t"`。
- 宽格式 HPLC 表（行 = 碳源、列 = 代谢物）可以先用
  `gemfitcom.io.hplc.hplc_wide_to_long` 转成长格式再加载。
- GEM 是 COBRA 兼容的 SBML 文件（`*.xml`）—— 通常是 AGORA2、
  CarveMe，或人工 curated 的模型。

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

## 真实数据示例

上面的合成 `toy_acetogen` 数据集刻意做得很小，目的是让 quickstart
几秒就能跑完、好排查。如果你想看在同样的长格式 schema 下
*真实*的实验数据长什么样，仓库还自带一份从
[GEMs-butyrate](https://github.com/Lishijiagg/GEMs-butyrate) 项目切出来的子集：
*Bifidobacterium longum* subsp. *infantis* ATCC 15697 在 GMC 对照
碳源上生长（3 个 OD replicate × 约 285 个时间点，加 6 种代谢物的
HPLC 端点测量）。

拉取数据 + 上游 SBML 模型（约 2.6 MB，没入库）：

```bash
python scripts/fetch_realistic_example.py
```

会在 `data/examples/realistic/` 下生成三个文件：

- `B_infantis_GMC_od.csv`   — 长格式 OD 生长曲线
- `B_infantis_GMC_hplc.csv` — 长格式 HPLC 端点测量，6 种代谢物
- `B_infantis_GEM.xml`      — 来自上游仓库的 SBML 模型

把这两个 CSV 当作**自己数据的模板**：复制列布局、替换数值，再让
strain config 指向它们。这个 realistic example 主要是**数据格式
参考**；要想端到端跑通完整 fit，还需要为 GMC 选一个交换反应 ID
（上游论文用自定义运输反应处理 HMO / 混合底物），超出了这份文档
入门篇的范围。
