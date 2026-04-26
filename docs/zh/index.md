# GemFitCom

**GEM + Fit + Community** —— 一个 Python pipeline，用体外生长和代谢物数据
校准肠道菌株的基因组尺度代谢模型（GEMs），并模拟多菌株代谢交互。

!!! warning "状态：v0.1"
    正在积极开发中。API 尚未稳定，如需复现请固定到具体 commit。

## 它能做什么

1. **单菌株 GEM 校准**，基于 OD 生长曲线和 HPLC 代谢物测量值：
    - 自动检测并添加缺失反应（针对 AGORA2 或 CarveMe 来源的模型），
      使用产物→反应的知识库（KB）。
    - 通过全局优化（差分进化）+ 局部网格精修，拟合底物交换反应的
      Michaelis–Menten 动力学参数（V<sub>max</sub>、K<sub>m</sub>）。
    - 用拟合后的动力学约束交换通量上下界。
2. **多菌株群落模拟**，三种模式：
    - `sequential_dfba` —— 每个时间步对每个菌株独立做 FBA，共享同一个代谢物池。
    - `micom` —— MICOM 稳态合作权衡（cooperative tradeoff）群落 FBA。
    - `fusion`（dMICOM）—— 动态模拟，每个时间步用 MICOM 合作权衡优化
      代替逐物种 FBA。
3. **交互分析** —— 从模拟通量轨迹中导出交叉饲喂（cross-feeding）和
   竞争（competition）矩阵 / 网络。

## 从哪开始

- 第一次用？先看 [安装](install.md)，再看 [快速开始](quickstart.md)。
- 想了解科学原理？看 [方法学](methods.md)。
- 找具体函数？看 [API 参考](api/io.md)（详情自动从 docstring 抓取，目前为英文）。

## 项目结构

```
src/gemfitcom/
├── io/           # 数据加载（OD、HPLC、SBML、YAML 配置）
├── preprocess/   # 生长率 / 滞后期提取，HPLC 数据清洗
├── medium/       # 培养基组成注册表（YCFA、LB、M9、...）
├── gapfill/      # 缺失反应检测与添加
├── kinetics/     # MM 动力学拟合（DE + 网格精修）
├── simulate/     # mono dFBA、sequential dFBA、MICOM、fusion (dMICOM)
├── interactions/ # 交叉饲喂 / 竞争网络构建
├── viz/          # 绘图工具
├── utils/        # 共用辅助（solver 自动检测等）
└── cli.py        # Typer CLI
```

## 许可证

MIT —— 见 [LICENSE](https://github.com/Lishijiagg/gemfitcom/blob/main/LICENSE)。
