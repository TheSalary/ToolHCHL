# IH-PromptDSI

> **Incremental Hierarchical Prompt-based Domain-Specific Intelligence**
> 基于增量层次化提示的领域专用智能工具检索与路由框架

本项目实现了一种面向 **工具检索（Tool Retrieval）** 与 **路由（Tool Routing）** 的增量学习框架 IH-PromptDSI，核心基于：
- **双流 Router**：语义流 + 依赖流融合
- **层次化 Box 嵌入空间**：L1 领域盒 + L2 意图盒
- **软提示池（Soft Prompt Pool）**：冻结 LLM + 可学习软提示
- **增量学习策略**：物理半冻结 + 经验回放 + 权重继承

支持 Base 训练及 Task1/Task2/Task3 三个增量任务，并提供 11 种消融实验配置。

---

## 目录

- [目录结构](#目录结构)
- [安装](#安装)
- [配置](#配置)
- [数据准备](#数据准备)
- [训练](#训练)
- [评估](#评估)
- [消融实验](#消融实验)
- [增量学习流程](#增量学习流程)
- [核心概念](#核心概念)

---

## 目录结构

```
IH-PromptDSI/
├── config.py                    # ⚠️ 唯一配置来源（所有路径/超参均在此定义）
├── run.py                       # 主入口：base + 增量任务训练/评估
├── run_ablation.py              # 消融实验入口
├── requirements.txt             # Python 依赖
├── .gitignore                   # Git 忽略规则
├── .cursorrules                 # AI Coding 规范

├── models/                      # 模型定义
│   ├── router.py                # 双流 Router（DualStreamRouter / FlatDualStreamRouter / SimpleLinearRouter）
│   ├── llm_caller.py            # LLMCaller 家族（Base / CL / NoPrompt）
│   ├── llm_wrapper.py           # SoftPromptPool + GenerativeToolCaller
│   ├── box_space.py             # 层次化 Box 嵌入空间（L1/L2）
│   └── build_real_box_space.py   # 构建 Base 初始 Box 空间脚本

├── training/                    # 训练逻辑
│   ├── trainer.py               # Base 训练器（IH_Trainer）
│   ├── trainer_cl.py            # CL 增量训练器（半冻结 + 旧盒备份）
│   ├── losses.py                # IHLoss（Task CE + Geo Loss + Contrastive Loss）
│   ├── ablation_trainer.py      # 统一消融训练器
│   ├── ablation_components.py   # 推理时消融组件（AblationRouter / AblationLoss）
│   ├── ablation_evaluator.py    # 统一评估器
│   └── continual.py             # 增量学习管理器（占位）

├── data_process/               # 数据处理
│   ├── dataset.py              # Base 数据集（IH_ToolDataset）
│   ├── dataset_cl.py           # CL 增量数据集（混合训练 + 经验回放）
│   ├── dataset_factory.py      # DataLoader 工厂
│   ├── preprocessor.py         # ToolBench 数据向量化与聚类
│   ├── expand_toolspace.py     # Box 空间扩展（合并新任务工具）
│   ├── expand_mekf.py          # m_global 扩展（占位）
│   └── process_toolbench/       # ToolBench 原始数据处理脚本
│       ├── preprocess1.py       # 提取工具信息
│       ├── preprocess2.py       # 清洗并匹配 L1 类别
│       ├── build_training_dataset.py  # 构建 85% 训练子集
│       ├── build_markov_matrix.py     # 构建 m_global Markov 转移矩阵
│       └── rewrite_intents_with_llama.py  # LLaMA 重写 API 描述

├── scripts/                    # 数据构建脚本
│   ├── prepare_task_data.py     # ⚠️ 主数据准备脚本（task1/2/3 均用此脚本）
│   ├── rebuild_base_clusters.py # 重建 Base 层次聚类
│   ├── prepare_flat_clusters.py # 生成 flat_clusters（w/o_hierarchy 消融用）
│   ├── extract_features.py      # 提取 Llama Layer-1 隐状态
│   ├── train_router_offline.py  # 离线 Router 训练
│   ├── extract_task1_tools.py  # 从记忆化数据提取 task1 工具
│   ├── prepare_task1_retrieval.py  # 映射 task1 检索数据到全局工具 ID
│   └── fix_task1_pipeline.py   # 修复 task1 数据流程

└── data/                       # ⚠️ 不上传 Git（由 .gitignore 忽略）
    ├── train/
    │   ├── clusters/            # Base 层次化空间（l1/l2 centers+widths, m_global, tool_to_l2, l2_to_l1）
    │   ├── raw/                 # Base 原始数据 + train_tools_with_id.json
    │   └── cache/               # Base 训练缓存
    ├── task1/
    │   ├── clusters/            # task1 层次化空间（继承 base l1，新 L2 由 HDBSCAN 发现）
    │   ├── flat_clusters/       # flat_space（w/o_hierarchy 消融用）
    │   ├── features/            # task1 工具 SentenceTransformer 特征
    │   ├── raw/                 # task1 原始数据
    │   └── cache/               # task1 训练缓存
    ├── task2/                   # 同 task1
    └── task3/                   # 同 task1
```

---

## 安装

### 环境要求

- Python ≥ 3.10
- CUDA ≥ 12.1（用于 Llama-3 训练）
- 显存 ≥ 24GB（单卡 8B 模型）

### 安装步骤

```bash
# 1. 创建 conda 环境
conda create -n toolprompt python=3.10 -y
conda activate toolprompt

# 2. 安装 PyTorch（根据你的 CUDA 版本选择命令）
# CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8:
# pip install torch --index-url https://download.pytorch.org/whl/cu118

# 3. 安装项目依赖
pip install -r requirements.txt

# 4. 下载 Llama-3 8B 模型（放置到配置路径）
# 默认路径: /data/wyx/llama3-8b
# 或修改 config.py 中的 LLAMA_PATH
```

### 目录初始化

```bash
# 构建 Base 初始 Box 空间（只需执行一次）
python models/build_real_box_space.py
```

---

## 配置

**所有配置集中在 `config.py`**，严禁在其他脚本中重复定义。

### Llama 模型路径

```python
# config.py 第 23 行
LLAMA_PATH = "/data/wyx/llama3-8b"   # 修改为你的 Llama-3 本地路径
```

### 任务配置（`TASK_CONFIGS`）

| 任务 | 说明 | 旧 Checkpoint | 工具数 |
|------|------|--------------|--------|
| `base` | 初始基础训练（11,112 工具） | 无 | 11,112 |
| `task1` | 增量任务 1 | checkpoints/base/*.pt | 1,640 新 |
| `task2` | 增量任务 2 | checkpoints/task1/*.pt | 640 新 |
| `task3` | 增量任务 3 | checkpoints/task2/*.pt | 643 新 |

### 消融实验配置（`ABLATION_CONFIGS`）

11 种消融实验，见 [消融实验章节](#消融实验)。

### 辅助函数

```python
from config import (
    get_task_cfg,          # 获取任务配置
    get_prev_task_cfg,     # 获取上一任务配置
    get_ablation_cfg,      # 获取消融实验配置
    get_checkpoint_dir,    # 获取 checkpoint 目录
    normalize,             # 规范化工具名称
    load_data_mappings,    # 加载工具 ID 映射
)
```

---

## 数据准备

> Base 数据（ToolBench）需要提前准备好放到 `data/train/raw/`。
> Task1/2/3 数据由 `scripts/prepare_task_data.py` 自动处理。

### 基础数据目录结构

```
data/train/raw/
├── memorization_train.json    # ToolBench 记忆化数据
├── memorization_eval.json     # ToolBench 记忆化评估数据
├── retrieval_train.json       # 检索训练数据（query + tool intent）
├── retrieval_eval.json        # 检索评估数据
├── train_tools_with_id.json   # 工具 ID 字典
└── tools.txt                  # 工具列表

# 以下由 scripts/prepare_task_data.py 自动生成：
├── clusters/
│   ├── l2_centers.pt         # L2 意图盒子中心 (582 × 384)
│   ├── l2_widths.pt          # L2 盒子宽度
│   ├── l1_centers.pt         # L1 领域盒子中心 (50 × 384)
│   ├── l1_widths.pt          # L1 盒子宽度
│   ├── m_global.pt           # Markov 转移矩阵 (num_tools × num_boxes)
│   ├── tool_to_l2.pt         # 工具 → L2 盒子映射
│   └── l2_to_l1.pt          # L2 → L1 领域映射
└── cache/                     # 训练样本缓存
```

### Base 空间构建（只需一次）

```bash
# 1. 下载/放置 ToolBench 数据到 data/train/raw/

# 2. 构建 Base 层次化 Box 空间
python models/build_real_box_space.py
```

### Task 数据准备（task1 / task2 / task3）

```bash
# 通用：scripts/prepare_task_data.py 支持 --stage 分阶段执行
python scripts/prepare_task_data.py --task task1 --stage 0   # 检查依赖文件
python scripts/prepare_task_data.py --task task1 --stage 1   # 提取新工具
python scripts/prepare_task_data.py --task task1 --stage 2   # 规范化检索数据
python scripts/prepare_task_data.py --task task1 --stage 3   # HDBSCAN 聚类 + 空间缝合

# task2 同理
python scripts/prepare_task_data.py --task task2 --stage 1
python scripts/prepare_task_data.py --task task2 --stage 2
python scripts/prepare_task_data.py --task task2 --stage 3

# task3 同理
python scripts/prepare_task_data.py --task task3 --stage 1
python scripts/prepare_task_data.py --task task3 --stage 2
python scripts/prepare_task_data.py --task task3 --stage 3
```

`--stage 3` 完成后，`data/<task>/clusters/` 下将生成所有 Box 空间文件。

---

## 训练

### Base 训练

```bash
# 单次完整流程（prepare + train + eval）
python run.py --task base --mode all

# 仅训练
python run.py --task base --mode train

# 指定 GPU
python run.py --task base --mode all --gpu 0,1,2,3

# 指定 GPU 数量
python run.py --task base --mode all --num-gpus 4
```

### 增量任务训练（task1 / task2 / task3）

```bash
# task1: 从 base checkpoint 继承权重
python run.py --task task1 --mode all

# task2: 从 task1 checkpoint 继承权重
python run.py --task task2 --mode all

# task3: 从 task2 checkpoint 继承权重
python run.py --task task3 --mode all
```

增量训练时会自动：
1. 加载上一任务 checkpoint，备份旧 Box 空间
2. 构建增量 Box 空间（继承旧 L1，新增 L2）
3. 冻结旧 Router 参数（半冻结）
4. 混合当前任务数据 + 经验回放数据

### 批量训练所有任务

```bash
python run.py --task all --mode all
```

### 常用参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--task` | `all` | `base` / `task1` / `task2` / `task3` / `all` |
| `--mode` | `all` | `prepare` / `train` / `eval` / `all` |
| `--stage` | `3` | prepare 阶段（仅 `--mode prepare` 时生效）|
| `--gpu` | `None` | GPU ID 列表，如 `"0,1"` |
| `--num-gpus` | `None` | 自动使用 N 张 GPU |
| `--old-checkpoint` | `None` | 手动指定旧 checkpoint 路径（覆盖 config.py）|

---

## 评估

### 评估已训练模型

```bash
# Base 评估
python run.py --task base --mode eval

# Task1 评估
python run.py --task task1 --mode eval

# 限制评估样本数
python run.py --task task1 --mode eval --limit 1000

# 多任务联合评估
python run.py --task task1 --mode eval --eval-tasks base,task1
```

---

## 消融实验

消融实验通过 `run_ablation.py` 驱动。

### 训练类消融（需要重新训练）

| 实验名 | 说明 | Router 类型 |
|--------|------|------------|
| `semi_freeze_off` | D2 关闭物理半冻结（全参数微调）| DualStreamRouter |
| `weight_inherit_off` | D3 关闭权重继承（随机初始化）| DualStreamRouter |
| `replay_off` | D4 关闭经验回放 | DualStreamRouter |
| `geo_loss_off` | E1 关闭几何包含损失 $L_{geo}$ | DualStreamRouter |
| `w/o_hierarchy` | E2 去掉 L1 层，仅用 L2 软距离盒子 | FlatDualStreamRouter |
| `flat_space` | E3 使用单层扁平空间 | FlatDualStreamRouter |
| `linear_router` | E4 无 Box 机制，纯线性投影 | SimpleLinearRouter |

```bash
# 示例：训练 semi_freeze_off 消融（task1）
python run_ablation.py --task task1 --mode all --ablation semi_freeze_off

# 示例：训练 geo_loss_off 消融（task2）
python run_ablation.py --task task2 --mode all --ablation geo_loss_off
```

### 推理类消融（无需重新训练，直接在已训好模型上评估）

| 实验名 | 说明 |
|--------|------|
| `router_semantic` | R1 仅语义流，去除依赖流和门控（固定 α=1.0）|
| `router_dependency` | R2 仅依赖流，去除语义流（固定 α=0.0）|
| `router_no_gate` | R3 去除门控，固定 α=0.5 融合 |
| `ablation_compare` | 综合对比：一次性运行 R1/R2/R3 并汇总表格 |

```bash
# 推理时消融评估（task1，已训练好）
python run_ablation.py --task task1 --mode eval --ablation router_semantic
python run_ablation.py --task task1 --mode eval --ablation router_dependency
python run_ablation.py --task task1 --mode eval --ablation router_no_gate

# 综合对比
python run_ablation.py --task task1 --mode eval --ablation ablation_compare
```

### 消融实验常用参数

| 参数 | 说明 |
|------|------|
| `--ablation` | 消融实验名（见上表）|
| `--train-only` | 仅训练，跳过评估 |
| `--eval-only` | 仅评估，跳过训练 |
| `--replay-per-tool` | 经验回放每工具采样数（默认 1）|
| `--resume <path>` | 从指定 checkpoint 恢复训练 |
| `--space-type` | `hierarchical`（默认）或 `flat` |

---

## 增量学习流程

### 完整流程示例（task1）

```
┌─────────────────────────────────────────────┐
│ Step 1: 数据准备                              │
│ python scripts/prepare_task_data.py \        │
│     --task task1 --stage 1                   │
│ python scripts/prepare_task_data.py \        │
│     --task task1 --stage 2                   │
│ python scripts/prepare_task_data.py \        │
│     --task task1 --stage 3                   │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│ Step 2: Base 训练（已有 checkpoint 可跳过）     │
│ python run.py --task base --mode all         │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│ Step 3: task1 增量训练                        │
│ # config.py 已配置 --old-checkpoint 指向     │
│ # checkpoints/base/*.pt，无需手动指定          │
│ python run.py --task task1 --mode all        │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│ Step 4: task1 评估                           │
│ python run.py --task task1 --mode eval       │
│                                          │
│ # 推理时消融（无需重新训练）                   │
│ python run_ablation.py \                   │
│     --task task1 --mode eval \              │
│     --ablation ablation_compare             │
└─────────────────────────────────────────────┘
```

### 后续增量任务（task2 / task3）

task2 → 继承 task1 checkpoint
task3 → 继承 task2 checkpoint

配置已在 `config.py` 的 `TASK_CONFIGS` 中自动关联，无需手动指定 `--old-checkpoint`。

---

## 核心概念

### 双流 Router（DualStreamRouter）

```
用户 Query
    │
    ├──► 语义流：softmax(-||q - l2_center||²)        （发现候选工具意图）
    │
    └──► 依赖流：m_global[t_prev]                   （利用上一工具 Markov 转移）
             │
         Gate MLP(α)
             │
    α × 语义得分 + (1-α) × 依赖得分
             │
         Top-K 选择
```

### 层次化 Box 嵌入空间

```
L1: 领域盒（Domain Box）—— 按功能领域粗分（50 个）
    │
    └──► L2: 意图盒（Intent Box）—— HDBSCAN 细粒度聚类发现（~582 个 base）
              │
          m_global: Markov 转移矩阵 [num_tools × num_boxes]
          刻画工具间调用转移概率
```

### 训练损失

| 损失 | 公式 | 作用 |
|------|------|------|
| Task Loss | $L_{task} = CE(p_{router}, y_{tool})$ | 工具分类 |
| Geo Loss | $L_{geo} = \sum \text{softplus}(d_{outside})$ | L2 盒须包含在 L1 盒内 |
| Contrastive Loss | NLL on router softmax | 拉近 router 输出与 box 距离 |

### 增量学习策略

| 策略 | 说明 |
|------|------|
| **半冻结（Semi-Freeze）** | 仅训练新 L2 盒子，旧 Router 参数冻结，防止遗忘 |
| **经验回放（Experience Replay）** | 混合当前任务数据 + 历史任务回放数据 |
| **权重继承（Weight Inheritance）** | 从上一任务 checkpoint 加载旧 Router 权重 |

---

## 项目规范

本项目遵循 `.cursorrules` 中的 Karpathy 风格规范：

1. **编码前先思考** — 明确假设，不确定就问
2. **简洁优先** — 不做需求之外的功能和抽象
3. **精准修改** — 只改必须改的，保持现有风格
4. **目标驱动** — 定义成功标准，循环验证
