"""
config.py
=========
IH-PromptDSI 全局配置（唯一真实来源）。

所有入口脚本（run.py / run_ablation.py）的路径、超参、消融实验配置
均从此文件导入，严禁在其他脚本中重复定义。

使用方式：
    from config import TASK_CONFIGS, ABLATION_CONFIGS, LLAMA_PATH, get_task_cfg

辅助函数：
    from config import normalize, load_data_mappings, get_checkpoint_dir
"""

import os
import re
import json

# ============================================================================
# 1. 路径配置
# ============================================================================
LLAMA_PATH = "/data/wyx/llama3-8b"

# ============================================================================
# 2. 任务配置（base / task1 / task2 / task3）
# ============================================================================
TASK_CONFIGS = {
    "base": {
        "name": "Base",
        "prev_task": None,

        # 聚类文件路径
        "cluster_dir": "./data/train/clusters",
        "l2_centers":  "./data/train/clusters/l2_centers.pt",
        "l2_widths":   "./data/train/clusters/l2_widths.pt",
        "l1_centers":  "./data/train/clusters/l1_centers.pt",
        "l1_widths":   "./data/train/clusters/l1_widths.pt",
        "m_global":    "./data/train/clusters/m_global.pt",
        "tool_to_l2":  "./data/train/clusters/tool_to_l2.pt",
        "l2_to_l1":    "./data/train/clusters/l2_to_l1.pt",

        # 数据文件路径
        "train_json":  "./data/train/raw/retrieval_train.json",
        "eval_json":   "./data/train/raw/retrieval_eval.json",
        "cache_dir":   "./data/train/cache",

        # 训练配置
        "old_checkpoint":   None,                                              # Base 无继承
        "old_num_tools":   11112,
        "old_num_boxes":   713,
        "epochs":          5,
        "batch_size":      16,
        "lr":              1e-4,
        "ckpt_dir":        "./checkpoints/base",
        "eval_limit":      5000,
    },

    "task1": {
        "name": "Task1",
        "prev_task": "base",

        "cluster_dir": "./data/task1/clusters",
        "l2_centers":  "./data/task1/clusters/l2_centers.pt",
        "l2_widths":   "./data/task1/clusters/l2_widths.pt",
        "l1_centers":  "./data/train/clusters/l1_centers.pt",
        "l1_widths":   "./data/train/clusters/l1_widths.pt",
        "m_global":    "./data/task1/clusters/m_global.pt",
        "tool_to_l2":  "./data/task1/clusters/tool_to_l2.pt",
        "l2_to_l1":    "./data/task1/clusters/l2_to_l1.pt",

        "train_json":  "./data/task1/raw/retrieval_train.json",
        "eval_json":   "./data/task1/raw/retrieval_eval.json",
        "cache_dir":   "./data/task1/cache",

        # CL 数据集专用
        "new_tools_json":       "./data/task1/clusters/task1_tools_with_id.json",
        "prev_task_tools_json": None,        # Task1 无更早任务
        "prev_task_train_json": None,

        "old_checkpoint": "./checkpoints/base/ih_prompt_dsi_cls_epoch_2_20260511_021554.pt",
        "old_num_tools":  11752,
        "old_num_boxes":  607,

        "epochs":     5,
        "batch_size": 16,
        "lr":         1e-3,
        "ckpt_dir":   "./checkpoints/task1",
        "eval_limit": 5000,
    },

    "task2": {
        "name": "Task2",
        "prev_task": "task1",

        "cluster_dir": "./data/task2/clusters",
        "l2_centers":  "./data/task2/clusters/l2_centers.pt",
        "l2_widths":   "./data/task2/clusters/l2_widths.pt",
        "l1_centers":  "./data/task1/clusters/l1_centers.pt",
        "l1_widths":   "./data/task1/clusters/l1_widths.pt",
        "m_global":    "./data/task2/clusters/m_global.pt",
        "tool_to_l2":  "./data/task2/clusters/tool_to_l2.pt",
        "l2_to_l1":    "./data/task2/clusters/l2_to_l1.pt",

        "train_json":  "./data/task2/raw/retrieval_train.json",
        "eval_json":   "./data/task2/raw/retrieval_eval.json",
        "cache_dir":   "./data/task2/cache",

        "new_tools_json":       "./data/task2/clusters/task2_tools_with_id.json",
        "prev_task_tools_json": "./data/task1/clusters/task1_tools_with_id.json",
        "prev_task_train_json": "./data/task1/raw/retrieval_train.json",

        "old_checkpoint": "./checkpoints/task1/ih_prompt_dsi_cls_best_epoch_1_20260511_032306.pt",
        "old_num_tools":  12392,
        "old_num_boxes":  626,

        "epochs":     5,
        "batch_size": 16,
        "lr":         1e-3,
        "ckpt_dir":   "./checkpoints/task2",
        "eval_limit": 5000,
    },

    "task3": {
        "name": "Task3",
        "prev_task": "task2",

        "cluster_dir": "./data/task3/clusters",
        "l2_centers":  "./data/task3/clusters/l2_centers.pt",
        "l2_widths":   "./data/task3/clusters/l2_widths.pt",
        "l1_centers":  "./data/task2/clusters/l1_centers.pt",
        "l1_widths":   "./data/task2/clusters/l1_widths.pt",
        "m_global":    "./data/task3/clusters/m_global.pt",
        "tool_to_l2":  "./data/task3/clusters/tool_to_l2.pt",
        "l2_to_l1":    "./data/task3/clusters/l2_to_l1.pt",

        "train_json":  "./data/task3/raw/retrieval_train.json",
        "eval_json":   "./data/task3/raw/retrieval_eval.json",
        "cache_dir":   "./data/task3/cache",

        "new_tools_json":       "./data/task3/clusters/task3_tools_with_id.json",
        "prev_task_tools_json": "./data/task2/clusters/task2_tools_with_id.json",
        "prev_task_train_json": "./data/task2/raw/retrieval_train.json",

        "old_checkpoint": "./checkpoints/task2/ih_prompt_dsi_cls_best_epoch_10_20260402_090127.pt",
        "old_num_tools":  13035,
        "old_num_boxes":  650,

        "epochs":     5,
        "batch_size": 16,
        "lr":         1e-3,
        "ckpt_dir":   "./checkpoints/task3",
        "eval_limit": 5000,
    },
}


# ============================================================================
# 3. 消融实验配置
# ============================================================================
# train_requires: 该实验是否需要重新训练（True=需训练，False=仅推理）
# router_cls   : 对应的 Router 类名（供训练/评估时动态选择）
# desc         : 简短描述（用于日志和文件命名）
ABLATION_CONFIGS = {
    # ----- 需要重新训练的消融实验 -----
    "semi_freeze_off": {
        "train_requires": True,
        "router_cls":    "DualStreamRouter",
        "desc":          "D2 关闭物理半冻结（全参数微调）",
    },
    "weight_inherit_off": {
        "train_requires": True,
        "router_cls":    "DualStreamRouter",
        "desc":          "D3 关闭权重继承（随机初始化）",
    },
    "replay_off": {
        "train_requires": True,
        "router_cls":    "DualStreamRouter",
        "desc":          "D4 关闭经验回放",
    },
    "geo_loss_off": {
        "train_requires": True,
        "router_cls":    "DualStreamRouter",
        "desc":          "E1 关闭几何包含损失 L_geo",
    },
    "w/o_hierarchy": {
        "train_requires": True,
        "router_cls":    "FlatDualStreamRouter",
        "desc":          "E2 去掉 L1 层，仅用 L2 软距离盒子",
    },
    "flat_space": {
        "train_requires": True,
        "router_cls":    "FlatDualStreamRouter",
        "desc":          "E3 使用单层扁平空间（无 L1/L2 层级）",
    },
    "linear_router": {
        "train_requires": True,
        "router_cls":    "SimpleLinearRouter",
        "desc":          "E4 使用 SimpleLinearRouter（无 box 机制，纯线性投影）",
    },

    # ----- 仅推理时评估的消融实验 -----
    "router_semantic": {
        "train_requires": False,
        "router_cls":    "DualStreamRouter",
        "desc":          "R1 仅语义流，去除依赖流和门控",
    },
    "router_dependency": {
        "train_requires": False,
        "router_cls":    "DualStreamRouter",
        "desc":          "R2 仅依赖流，去除语义流",
    },
    "router_no_gate": {
        "train_requires": False,
        "router_cls":    "DualStreamRouter",
        "desc":          "R3 去除门控，固定 alpha=0.5 融合",
    },
    "ablation_compare": {
        "train_requires": False,
        "router_cls":    "DualStreamRouter",
        "desc":          "综合对比：推理时消融对比所有 R 系列实验",
    },
}


# ============================================================================
# 4. 推理时消融实验集合（用于快速判断）
# ============================================================================
INFERENCE_ABLATIONS = {
    name for name, cfg in ABLATION_CONFIGS.items()
    if not cfg["train_requires"]
}


# ============================================================================
# 5. 辅助函数
# ============================================================================
def get_task_cfg(task_name: str) -> dict:
    """根据任务名返回配置，找不到则抛出 KeyError（key 大小写不敏感）。"""
    key = task_name.lower()
    if key not in TASK_CONFIGS:
        raise KeyError(f"未知任务: {task_name!r}，可选: {list(TASK_CONFIGS.keys())}")
    return TASK_CONFIGS[key]


def get_prev_task_cfg(task_name: str) -> dict | None:
    """根据任务名返回上一任务的配置（base 返回 None）。"""
    cfg = get_task_cfg(task_name)
    prev = cfg.get("prev_task")
    return get_task_cfg(prev) if prev else None


def get_ablation_cfg(ablation_name: str) -> dict | None:
    """根据消融实验名返回配置，找不到返回 None。"""
    return ABLATION_CONFIGS.get(ablation_name)


def get_all_task_names() -> list[str]:
    """返回所有任务名（不含 'all'）。"""
    return list(TASK_CONFIGS.keys())


def get_all_ablation_names() -> list[str]:
    """返回所有消融实验名。"""
    return list(ABLATION_CONFIGS.keys())


def get_checkpoint_dir(task_name: str, ablation: str | None = None) -> str:
    """
    根据任务名和消融实验名返回 checkpoint 目录。
    消融实验的 checkpoint 统一放在 <ckpt_dir>/ablation/<ablation_name> 下。

    task_name 可以是 "task1" / "Task1" / "base" / "Base" 等（大小写不敏感）。
    """
    # 规范化：支持 "Task1" / "base" 等大小写变体
    key = task_name.lower()
    cfg = get_task_cfg(key)
    base = cfg["ckpt_dir"]
    if ablation is None or ablation == "ablation_compare":
        return base

    # 统一映射：w/o_hierarchy → wo_hierarchy（避免路径含斜杠问题）
    ab_subdirs = {
        "semi_freeze_off":    "ablation/semi_freeze_off",
        "weight_inherit_off": "ablation/weight_inherit_off",
        "replay_off":         "ablation/replay_off",
        "geo_loss_off":       "ablation/geo_loss_off",
        "w/o_hierarchy":      "ablation/wo_hierarchy",
        "flat_space":         "ablation/flat_space",
        "linear_router":      "ablation/linear_router",
    }
    sub = ab_subdirs.get(ablation, f"ablation/{ablation}")
    return f"{base}/{sub}"


# ============================================================================
# 6. 工具函数（供所有入口脚本共享）
# ============================================================================
def normalize(name: str) -> str:
    """将工具/接口名称规范化为小写字母数字串，用于 ID 映射。"""
    return re.sub(r'[^a-zA-Z0-9]', '', str(name)).lower()


def load_data_mappings(task_cfg: dict) -> dict[str, int]:
    """
    加载工具 ID <-> 名称的完整映射（支持 Base + 新任务工具）。
    返回 {"api_name_for_tool_name": tool_id} 字典。
    """
    name_to_id: dict[str, int] = {}

    # Base 工具字典
    base_tools_path = "./data/train/raw/train_tools_with_id.json"
    if os.path.exists(base_tools_path):
        with open(base_tools_path, "r", encoding="utf-8") as f:
            base_tools = json.load(f)
        for t in base_tools:
            api_norm = normalize(t.get("api_name", ""))
            tool_norm = normalize(t.get("tool_name", ""))
            name_to_id[f"{api_norm}for{tool_norm}"] = t.get("tool_id")

    # 新任务工具字典（如果存在）
    task_name = task_cfg.get("name", "").lower()
    for prefix in ["task1", "task2", "task3"]:
        if prefix in task_name or task_cfg.get("cluster_dir", "").startswith(f"./data/{prefix}"):
            for n in ["task1", "task2", "task3"]:
                tools_json = os.path.join(f"./data/{n}/clusters", f"{n}_tools_with_id.json")
                if os.path.exists(tools_json):
                    with open(tools_json, "r", encoding="utf-8") as f:
                        new_tools = json.load(f)
                    for t_id_str, text in new_tools.items():
                        if isinstance(text, str):
                            a_match = re.search(r'API:\s*(.*?)\.(?:\s*API Description:|$)', text)
                            t_match = re.search(r'Tool:\s*(.*?)\.(?:\s*Description:|\s*API:)', text)
                            if a_match and t_match:
                                a_name = a_match.group(1).strip()
                                t_name = t_match.group(1).strip()
                                name_to_id[normalize(f"{a_name}for{t_name}")] = int(t_id_str)
    return name_to_id
