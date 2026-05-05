# Spatial dFBA v0.1 设计文档

| Field | Value |
|---|---|
| Date | 2026-05-05 |
| Author | Shijia Li |
| Status | Approved (pending implementation) |
| Target | `gemfitcom` v0.x（新增 `spatial` 子模块） |
| Estimated effort | 10-13 周（按 1-2 小时/天节奏） |

---

## 1. 背景与目标

### 1.1 问题陈述

`gemfitcom` v01 已经有三种 well-mixed 社区模拟模式（`sequential_dfba` / `micom` / `fusion`），但都假设培养基瞬时均匀。许多重要的微生物组现象——尤其是肠道环境中的好氧/厌氧梯度、cross-feeding 的空间组织、菌群在膜面 vs 腔体的分布——**本质上是空间异质的**，well-mixed 假设丢失了关键物理。

### 1.2 设计目标（按优先级）

1. **B（首要）**：为用户自己的研究问题（如膳食干预下特定菌株群落的空间组织）提供可定制的空间模拟工具
2. **C**：作为 BacArena 的 Python 替代品对外发布，融入 cobra/MICOM 生态
3. **D**：教学 / 演示用途，能跑出可视化清晰说明空间因素的影响

### 1.3 参考文献与已有工具

- **BacArena**（R 包）：individual-based、2D 网格、agent-based dFBA。Python 生态没有同类工具
- **van Hoek & Merks 2017** "Multiscale Spatiotemporal Model... Infant Gut Microbiota"（C 实现）：continuum PDE × FBA 耦合，模拟婴儿肠道兼性好氧→专性厌氧菌演替

v0.1 走 **continuum PDE 路线**（参考 van Hoek & Merks），不走 individual-based。

### 1.4 非目标（Non-Goals）

明确**不做**的事情，避免 scope creep：

- 不做 individual-based / agent-based 模拟（v0.x 整代不做）
- 不做 2D / 3D 几何（v0.1 只 1D；2D 留给 v0.2）
- 不做 Web UI / 上传服务（v0.1 只 CLI + Python API；Web UI 留给 v0.2 选配）
- 不做 Boolean regulatory network、调控基因表达（始终走"FBA 自然约束 → emergent switch"）
- 不做 C/C++ 扩展（用 Python + numpy/scipy + 可选 numba）
- 不做对流（advection）项（留给 v0.3+）
- 不做 host-microbe interaction 的免疫细胞、上皮细胞动力学（mucosa 边界用简化的源项 BC）

---

## 2. 关键决策

每条决策都有**为什么**，避免后续质疑被翻盘。

| # | 决策 | 选择 | Why |
|---|---|---|---|
| 1 | 实现语言 | Python（不用 C/C++） | cobra/optlang 求解器本身已是 C++；NumPy/SciPy 扩散已向量化；Numba 可后置加速；维护成本远低于 C |
| 2 | 建模范式 | Continuum / PDE | 与目标 paper 一致；可扩展到大尺度；代码与 v01 dFBA 高度同构 |
| 3 | 几何 | 1D 肠道横截面（mucosa-lumen） | 1D 已能展示 aerobic→anaerobic 切换核心物理；2D 在 v0.1 浪费算力 |
| 4 | v0.1 范围 | 骨架版：2 菌 + 4 代谢物 + 1h 量级仿真 | 验证 PDE+FBA 耦合 pipeline，不追求科学完整性 |
| 5 | 与 v01 关系 | `src/gemfitcom/spatial/` 子模块 | 共享 io/medium/kinetics/viz；spatial 依赖作 optional extras |
| 6 | 输入方式 | CLI + Python API（文件路径） | 与 v01 一致；Web UI 作为 v0.2+ 选配 |
| 7 | PDE 求解 | 手写 operator splitting + scipy.ndimage.laplace | FiPy/py-pde 是重依赖且对非光滑反应项不友好；1D 扩散五行 numpy |
| 8 | 时间分裂 | Lie splitting（反应 → 扩散 → 边界） | Strang splitting 对 v0.1 dt 没必要；与 well-mixed 极限测试不冲突 |
| 9 | 反应 = dFBA | 每格、每菌、每步用当地浓度算 MM 上界，写入 cobra `lower_bound`，optimize | spatial dFBA 的标准做法；用户明确强调这是核心 |
| 10 | 性能 backend | strategy 模式（serial / joblib），加可选量化 cache | embarrassingly parallel；cobra 模型必须用进程池而非线程（非 thread-safe） |
| 11 | 数值精度 | 显式 FTCS 扩散 + `B *= exp(mu·dt)` 解析积分 | mass conservation + positivity 由不变量测试守住；隐式方法对 v0.1 过早 |
| 12 | 配置格式 | YAML + pydantic 校验 | 与 v01 一致；pydantic 比手写校验稳健 |
| 13 | 输出格式 | `.npz` 默认，`netcdf` 可选 | npz 零额外依赖；netcdf 需 xarray，作 optional |

---

## 3. 架构

### 3.1 核心抽象

```
SpatialState        ← 系统当前状态（dataclass）
  ├─ metabolites    : ndarray (n_mets, n_grid)         浓度场 mmol/L
  ├─ biomass        : ndarray (n_species, n_grid)      生物量场 gDW/L
  └─ t              : float                             当前时间 h

Geometry1D          ← 1D 网格 + mucosa-lumen 边界
  ├─ n_grid, dx
  ├─ boundary_sources(t) : 给定 t 返回每个边界节点的通量/Dirichlet 值
  └─ diffusion_op        : sparse Laplacian 矩阵（预编译）

ExchangeKinetics    ← 每菌 × 每交换反应的 (Vmax, Km, mode)
  └─ mm_upper_bound(C_local) → ndarray (n_exchanges,)

ReactionEngine      ← 每个网格点的 dFBA 编排
  ├─ models       : list[cobra.Model]   每个菌一个，已加 medium 骨架约束
  ├─ kinetics     : list[ExchangeKinetics]
  ├─ backend      : SerialBackend | JoblibBackend
  └─ step(C, B, dt) → (mu_field, flux_field)

Simulator           ← 时间循环编排
  ├─ state, geometry, reaction_engine, recorder
  └─ run(t_end)
```

### 3.2 单步算法

```python
# 反应子步（dFBA per grid cell，可并行）
mu, flux = engine.step(state.C, state.B, dt)

# 应用反应到 state
state.B *= np.exp(mu * dt)                              # 解析积分
state.C += np.einsum('jix,ix->jx', flux, state.B) * dt  # 通量 × 生物量
state.C  = np.maximum(state.C, 0)                       # 数值正性 clip

# 扩散子步（向量化，所有代谢物同时）
state.C += dt * D[:, None] * laplacian(state.C, geometry)

# 边界源项（mucosa: O₂ 注入；lumen: glucose 注入）
state.C += dt * geometry.boundary_sources(state.t)

state.t += dt
recorder.maybe_save(state)
```

**关键点**：
- dFBA 的"动态"体现在 `engine.step` 内部对 cobra `lower_bound` 的逐格重写
- Lie splitting 一阶精度，v0.1 dt = 0.1h 完全够；后续要更高阶可换 Strang
- 维度无关的 API：1D → 2D 只需把 grid_shape 从 `(50,)` 换成 `(50,50)`，扩散用 N 维 Laplacian，其他代码 0 改动

### 3.3 性能 backend

四层优化，可叠加：

| Layer | 描述 | 默认 | 适用场景 |
|---|---|---|---|
| 0 | 跳过空格（B < ε） | 永远开 | 任何场景 |
| 1 | 进程级并行（joblib） | n_grid ≥ 16 时自动开 | 中等规模以上 |
| 2 | 量化 FBA 结果缓存 | opt-in（`--cache`） | 浓度场平滑、相邻格相似时 |
| 3 | 自适应 FBA 子循环 | v0.2+ 再做 | — |

**joblib 实现要点**：
- 用 `loky` backend（跨平台、Win 友好）
- worker `initializer` 加载一次 cobra 模型，避免每次 pickle
- cobra 模型不 thread-safe，**必须**用进程池
- 单格 FBA < 16 时 fallback serial（进程开销 > 收益）

**cache 实现要点**：
- key = (species_id, quantize(local_C, tolerance))
- LRU，max_size 可配（默认 10000）
- 量化 tolerance 默认 1%；测试守住该容差下解的相对误差 < 2%

### 3.4 模块划分

#### 新增（`src/gemfitcom/spatial/`）

```
spatial/
├── __init__.py             # 公开 API
├── state.py                # SpatialState
├── geometry.py             # Geometry1D
├── diffusion.py            # 稀疏 Laplacian + 扩散步进
├── kinetics.py             # ExchangeKinetics + MM 上界
├── reaction.py             # ReactionEngine（高层）
├── backends.py             # SerialBackend / JoblibBackend
├── cache.py                # 量化 memoization
├── simulator.py            # Simulator + 时间循环
├── recorder.py             # SnapshotRecorder
├── config.py               # YAML schema（pydantic）
├── viz.py                  # kymograph、final-state、生物量演化曲线
└── cli.py                  # typer 子命令
```

#### 复用 v01 现有模块（不重写）

| spatial 调用 | v01 位置 |
|---|---|
| SBML 加载 | `gemfitcom.io.models.load_model` |
| 培养基定义 | `gemfitcom.medium.medium.Medium`、`.registry` |
| 培养基约束 | `gemfitcom.medium.constraints` |
| MM 公式 | `gemfitcom.kinetics.mm` |
| Vmax/Km 数据 | `gemfitcom.kinetics.fit` 输出 |
| CLI 框架 | `gemfitcom.cli` (typer app) |
| 进度反馈 | `gemfitcom.utils.progress` |
| 求解器选择 | `gemfitcom.utils.solver` |

**spatial 不放进 `gemfitcom/simulate/`**——它的 PDE × grid 数据模型与 sequential_dfba/micom/fusion 的 well-mixed × ODE 完全不同，独立子包边界更清晰。

### 3.5 依赖

`pyproject.toml` 新增 optional extras：

```toml
[project.optional-dependencies]
spatial = [
    "joblib>=1.3",          # 并行
    "pydantic>=2.5",        # config 校验
]
spatial-netcdf = [           # 可选：netcdf 输出
    "xarray>=2023.0",
    "netcdf4>=1.6",
]
```

---

## 4. 数据流

```
                ┌─ sim.yaml ────────┐
                │ geometry          │
                │ species[].gem ────┼──> cobra.io.read_sbml_model
                │ species[].kinetics┼──> ExchangeKinetics 加载
                │ metabolites       │
                │ backend           │
                └──────────┬────────┘
                           │
                           v
                  ┌──── load_config() ────┐
                  │  pydantic 校验         │
                  └──────────┬────────────┘
                             │
                             v
              ┌─────── build_simulator() ───────┐
              │  Geometry(n_grid, dx, BCs)      │
              │  ReactionEngine(models, kin)    │
              │     └─ medium.constraints 套上  │
              │  SpatialState(init_C, init_B)   │
              │  Recorder(output_dir, every)    │
              └─────────────┬───────────────────┘
                            │
                            v
                ┌───── Simulator.run() ─────┐
                │  for step in range(N):    │
                │    engine.step(C, B, dt)  │  ← 反应（并行）
                │    diffuse(C, dt)          │  ← 扩散
                │    apply_boundary(C, t)    │  ← 源项
                │    recorder.maybe_save()   │
                └─────────────┬─────────────┘
                              │
                              v
                ┌──── results/spatial_run/ ──┐
                │  config_resolved.yaml      │  审计
                │  snapshots/                │
                │    t=0.0.npz, t=1.0.npz... │
                │  metadata.json             │
                │  warnings.json             │
                │  log.txt                   │
                └────────────────────────────┘
```

中间产物全是文件：可审计、可重跑。

---

## 5. API 设计

### 5.1 三层 Python API

**高层（90% 用户）**：
```python
from gemfitcom.spatial import run

result = run("sim.yaml", output_dir="./results/run01")
# result: SimulationResult
```

**中层（自定义某个组件）**：
```python
from gemfitcom.spatial import (
    load_config, build_simulator, Simulator, Recorder
)

cfg = load_config("sim.yaml")
sim = build_simulator(cfg)
sim.recorder = Recorder(every=0.5)
sim.run(t_end=24.0)
```

**低层（写新算法、调试）**：
```python
from gemfitcom.spatial import (
    SpatialState, Geometry1D, ReactionEngine, diffuse_step
)

state = SpatialState.from_arrays(C=..., B=...)
geom = Geometry1D(n_grid=50, dx=2e-5, boundary={...})
engine = ReactionEngine(models=[m1, m2], kinetics=[k1, k2], backend="serial")

for step in range(240):
    mu, flux = engine.step(state.C, state.B, dt=0.1)
    state.B *= np.exp(mu * 0.1)
    state.C += flux_to_dC(flux, state.B, dt=0.1)
    diffuse_step(state.C, geom, dt=0.1)
    state.t += 0.1
```

三层基于同一组对象，不会出现"高层一套数据结构、低层另一套"。

### 5.2 CLI

```bash
gemfitcom spatial run CONFIG.yaml [--output DIR] [--n-jobs N] [--cache] [--dry-run]
gemfitcom spatial viz RESULT_DIR [--what kymograph|biomass|all] [--out FIG_DIR]
gemfitcom spatial validate CONFIG.yaml          # 只校验 config
```

`--dry-run`：加载所有 GEM、构建 Simulator、打印 estimated cost（FBA 调用次数 + 估时），但不跑——避免大仿真踩坑。

### 5.3 YAML config schema

```yaml
# === geometry ===
geometry:
  dim: 1
  n_grid: 50
  length: 1.0e-3               # 物理长度，米
  boundary:
    mucosa:                    # 网格点 0
      type: flux               # flux | dirichlet | reflecting
      sources:
        EX_o2_e:    1.0e-7     # mol/(m²·s)，host 分泌
    lumen:                     # 网格点 n_grid-1
      type: dirichlet
      values:
        EX_glc__D_e: 5.0       # mmol/L

# === time ===
simulation:
  t_end: 24.0                  # h
  dt: 0.1                      # h
  snapshot_every: 1.0          # h
  cfl_safety: 0.4              # 扩散 CFL 安全因子

# === species ===
species:
  - name: ecoli
    gem: ./gems/ecoli_core.xml
    biomass_reaction: BIOMASS_Ecoli_core_w_GAM   # 可选，缺省自动检测
    kinetics: ./kinetics/ecoli.yaml
    init:
      mode: uniform              # uniform | gaussian | step | from_array
      value: 1.0e-3              # gDW/L

  - name: fprau
    gem: ./gems/fprau_mini.xml
    kinetics: ./kinetics/fprau.yaml
    init:
      mode: gaussian
      center: 0.7                 # 相对位置 0~1（0=mucosa, 1=lumen）
      sigma: 0.1
      peak: 1.0e-3

# === metabolites tracked on grid ===
metabolites:
  - id: o2_e                       # cobra metabolite ID（BiGG）
    diffusion: 2.1e-9              # m²/s
    init: { mode: uniform, value: 0.21 }

  - id: glc__D_e
    diffusion: 6.7e-10
    init: { mode: uniform, value: 0.0 }

  - id: ac_e
    diffusion: 1.1e-9
    init: { mode: uniform, value: 0.0 }

  - id: but_e
    diffusion: 9.0e-10
    init: { mode: uniform, value: 0.0 }

# === performance backend ===
backend:
  type: auto                     # auto | serial | joblib
  n_jobs: -1
  cache:
    enabled: false
    tolerance: 0.01
    max_size: 10000

# === output ===
output:
  format: npz                    # npz | netcdf
  precision: float32             # snapshots 精度
```

### 5.4 Kinetics YAML（每菌一份）

```yaml
species: ecoli
exchanges:
  EX_o2_e:        { v_max: 15.0,  K_m: 0.005 }
  EX_glc__D_e:    { v_max: 10.0,  K_m: 0.5   }
  EX_ac_e:        { v_max: 5.0,   K_m: 0.1, mode: bidirectional }
  EX_but_e:       { v_max: 0.0,   K_m: 1.0 }
```

`mode`：`uptake_only`（默认）/ `bidirectional`（产生时上界用 MM）。

---

## 6. 错误处理

两档：

**启动期 fail-fast**——config / 模型 / CFL 检查
| 情形 | 行为 |
|---|---|
| config 字段缺失或类型错误 | pydantic 报错，列出所有问题 |
| 模型缺少 config 引用的交换反应 | 报错，列出缺失 reaction id |
| CFL 违反（dt > cfl_safety × dx²/(2D_max)） | 报错给建议 dt |
| 无可用 LP solver | 报错（沿用 v01 `utils.solver`） |

**运行期 warn-and-continue**——单格挂掉不能整轮挂
| 情形 | 行为 |
|---|---|
| FBA infeasible（局部死区） | mu=0、flux=0；记 `warnings.json`；不抛异常 |
| 浓度变负（数值噪声） | clip 到 0；如 \|负值\| > 1e-6 抛 RuntimeError（说明 dt 太大） |
| 生物量爆炸（exp 溢出） | mu × dt > 5 时 warn + clip mu |
| backend=joblib 但模型不能 pickle | 自动 fallback serial + warn |

运行期警告统一汇总到 `warnings.json`（不刷屏 stdout）。

---

## 7. 测试策略

### 7.1 测试金字塔

```
                    ┌──────────────┐
                    │  E2E（1）    │  示例 sim.yaml 端到端
                    ├──────────────┤
                    │  集成（4-6） │  多组件耦合 + 不变量
                    ├──────────────┤
                    │  单元（30+） │  每模块独立逻辑
                    └──────────────┘
```

### 7.2 单元测试

| 文件 | 覆盖 | 关键案例 |
|---|---|---|
| `test_state.py` | SpatialState | shape 校验、序列化往返 |
| `test_geometry.py` | Geometry1D | 边界节点、源项加和、坐标映射 |
| `test_diffusion.py` | 扩散步 | **解析对比**：高斯初值 vs `1/√(4πDt) exp(-x²/4Dt)`，t=0.1 误差 < 1% |
| `test_kinetics.py` | MM 上界 | 单调性、边界（C=0、C→∞）、负浓度防御 |
| `test_reaction_serial.py` | 单格 dFBA | 与裸 cobra 调用结果完全一致 |
| `test_cache.py` | 量化 memoization | key 命中、tolerance 边界 |
| `test_config.py` | YAML schema | 必填缺失、类型错误、CFL 检查 |
| `test_recorder.py` | 写盘 | 频率、文件名、读回无损 |
| `test_viz.py` | 可视化 | 输出 PNG、shape 正确（不验视觉） |

数值测试中 **FBA 被 mock**（`lambda C, B, dt: (zeros, zeros)`），保证 CI 速度。

### 7.3 集成测试

**`test_backends_consistency.py`** ★ 关键
```
result_serial = Simulator(backend="serial").run(...)
result_joblib = Simulator(backend="joblib").run(...)
result_cached = Simulator(backend="joblib", cache=True, tolerance=1e-3).run(...)

assert allclose(result_serial.B, result_joblib.B, rtol=1e-10)   # 浮点等价
assert allclose(result_serial.B, result_cached.B, rtol=2e-2)    # 量化容差
```

**`test_simulator_short.py`** — 10 步、3 网格、2 菌 smoke

### 7.4 不变量测试 ★

**`test_invariant_mass_conservation.py`**
- 闭边界 + 关反应 + 跑 1000 步纯扩散
- `∫C(x,t) dx` 守恒，误差 < 1e-10

**`test_invariant_positivity.py`**
- 任何配置：`C ≥ -1e-12` 永远成立、`B ≥ 0` 永远成立

**`test_invariant_well_mixed_limit.py`** ★★ 最重要
- D = 1e-3 m²/s（极大）→ 等价 well-mixed
- `mean(C)(t)`、`mean(B)(t)` 与 v01 `sequential_dfba` 误差 < 5%
- **同时验证**：扩散对 + 反应对 + 与 v01 接口语义一致

**`test_invariant_steady_state.py`**
- 单菌 + 持续供应 + t=100h
- 收敛到稳态（尾段 10h std 判定）

### 7.5 端到端

**`test_pipeline_e2e.py`**
- 跑 `examples/spatial/ecoli_fprau_skeleton/sim.yaml`，t_end = 1h
- 检查定性现象：
  - O₂：mucosa > lumen
  - F. prausnitzii biomass：lumen > mucosa
  - FBA infeasibility 格子 < 5%
- 跟 `expected_output/snapshot_t1h.npz` 对比（rtol=1e-3）

### 7.6 CI 性能预算

| 类别 | 单条上限 | 总耗时 |
|---|---|---|
| 单元 | 1 s | 30 s |
| 集成 | 5 s | 30 s |
| 不变量 | 10 s | 60 s |
| E2E | 30 s | 30 s |
| **总** | | **< 3 min** |

完整 24h 仿真标记 `@pytest.mark.slow`，CI 不跑。

### 7.7 算法选择 ↔ 验证不变量

| 算法 | 守住的不变量 |
|---|---|
| Lie splitting | well-mixed 极限测试 |
| 显式 FTCS 扩散 | mass conservation + positivity |
| MM 上界 + 显式更新 | 长期稳态测试 |
| `B *= exp(mu·dt)` | 不需要次小步长 |

---

## 8. 实现路线图

5 个独立可合并 PR。每个 PR 跑通 CI 才能合，不允许中间状态半成品。

### PR 1：基础设施 + 数值核心（无 FBA）

- `state.py`、`geometry.py`、`diffusion.py`、`recorder.py`、`config.py`（部分）
- 单元测试 + `test_invariant_mass_conservation` + `test_invariant_positivity`（FBA mock）
- pyproject.toml 加 spatial extras
- **验证**：纯扩散仿真能跑；mass conservation 误差 < 1e-10
- **工期（1-2h/d）**：1.5-2 周

### PR 2：kinetics + reaction + serial backend

- `kinetics.py`、`reaction.py`、`backends.py`（SerialBackend）
- 完整 `config.py`
- `examples/spatial/ecoli_fprau_skeleton/`
- 单元 + `test_simulator_short`
- **验证**：示例 sim.yaml 跑 10 步不报错
- **工期**：2.5-3.5 周（颗粒度大可拆 PR2a/PR2b）

### PR 3：simulator + 三层 API + CLI

- `simulator.py`、`__init__.py`、`cli.py`
- `test_invariant_well_mixed_limit` ★、`test_pipeline_e2e`、回归基准
- docs/spatial/quickstart.md
- **验证**：CLI 跑通示例；well-mixed 极限测试通过
- **工期**：2-3 周
- **里程碑**：合并后 v0.1 功能完整

### PR 4：joblib 并行 + cache

- `JoblibBackend`、`cache.py`
- `test_backends_consistency` ★
- `--n-jobs`、`--cache` CLI flags
- docs/spatial/performance.md
- **验证**：50 网格示例并行 3-5x 加速；consistency 测试通过
- **工期**：1.5-2.5 周

### PR 5：可视化 + 文档收尾

- `viz.py`：kymograph、final-state、生物量曲线
- `gemfitcom spatial viz` 子命令
- docs/spatial/ 完整、README 加段、24h demo 截图
- **验证**：viz 一键出图；docs build 无 warning
- **工期**：1.5-2.5 周

### 依赖关系

```
PR1 ──> PR2 ──> PR3 ──> PR4
                  └────> PR5
```

PR4 / PR5 在 PR3 后可并行。

### 累计

| 累计到 | 实际时间 |
|---|---|
| PR 1 | 1.5-2 周 |
| PR 2 | 4-5.5 周 |
| PR 3 | 6-8.5 周 |
| PR 4 | 7.5-11 周 |
| PR 5 | 9-13 周 |

**总：10-13 周（2.5-3 个月）按 1-2h/d 节奏**

PR 3 合并后已可用——可以开始 B 目标的科研探索，不必等所有 PR 完成。

---

## 9. 后续版本展望（非 v0.1 范围）

| 版本 | 主要工作 |
|---|---|
| v0.2 | 2D 几何（mucosa-lumen + 切向）；mucus 层；多菌 5-10 株；diet shift；adaptive FBA subcycling |
| v0.3 | 对流项（peristalsis）；advection-diffusion-reaction；上风格式 |
| v0.4 | Web UI（Streamlit / Gradio 本地浏览器上传）；批量参数扫描 |
| v0.5+ | Rust 扩展加速热点；GPU LP（如果到那一步还有性能瓶颈） |

---

## 10. 开放问题（实施期遇到再决定）

- pydantic v2 的 strict-mode 还是 lax-mode？倾向 strict
- `SimulationResult` 序列化是 dataclass + json 还是直接 npz dump？倾向后者（与现有 recorder 一致）
- 进度反馈：一个总体 tqdm bar 还是嵌套（snapshot bar + 子步 bar）？倾向单层
- Recorder 是否支持流式（边跑边写）vs 批量（end-of-run dump）？v0.1 走流式（容错）
