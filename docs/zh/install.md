# 安装

## 环境要求

- Python ≥ 3.10
- 一个 LP 求解器。GLPK（cobra 自带）开箱即用；CPLEX 是可选项，
  在大型群落模型上明显更快。

## 从源码安装

```bash
git clone https://github.com/Lishijiagg/gemfitcom
cd gemfitcom
pip install -e ".[dev]"
```

`dev` 这个 extra 会一起装上 `pytest`、`ruff` 和 `pre-commit`。
如果还要改文档，再追加 `".[docs]"` 安装 `mkdocs-material` 和 `mkdocstrings`。

## 验证安装

```bash
gemfitcom --help
gemfitcom solvers
```

`gemfitcom solvers` 会列出 cobra 在你系统上检测到的所有 LP 后端，
以及 GemFitCom 实际会选用的那个（优先级：CPLEX > Gurobi > GLPK）。
如果这一行是空的，说明安装不完整 —— 通常是因为该 solver 的
cobra Python bindings 没装。

## 启用 CPLEX

只装 IBM CPLEX Studio 是 **不够的**。cobra 需要 CPLEX 的 Python bindings，
它们随 CPLEX Studio 一起发布，但要单独安装：

```bash
cd "$CPLEX_STUDIO_DIR/python"
python setup.py install
```

用 `gemfitcom solvers` 验证 —— CPLEX 应该出现在 "Available" 列表里。

## 跑测试

```bash
pytest
```

`tests/test_cli.py` 里的端到端冒烟测试会用 `data/examples/` 下的合成数据集
跑完整的 CLI。大约耗时 20 秒，因为 `fit_kinetics` 的差分进化阶段是真跑的。
