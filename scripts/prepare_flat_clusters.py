#!/usr/bin/env python3
"""
scripts/prepare_flat_clusters.py
================================
为扁平化消融实验（w/o_hierarchy）生成 flat_clusters 目录。

逻辑：
  核心观察：m_global.pt 的 shape 已经是 [num_tools, num_boxes]，
  与 m_global_tool 完全相同！因此扁平空间的 m_global_tool.pt
  就是原有 m_global.pt 的直接复制。

  但是 L1 相关的文件（l1_centers、l1_widths、l2_to_l1）不再需要，
  我们重新生成一份不含 L1 依赖的 l2_centers / l2_widths / tool_to_l2。

  注意：l2_centers / l2_widths / tool_to_l2 与原 clusters/ 完全相同，
  扁平化只改变了 m_global（从 L1 级映射变为工具级直接映射）。

生成的文件结构：
  {task}/flat_clusters/
    ├── l2_centers.pt      # 与 clusters/ 相同
    ├── l2_widths.pt       # 与 clusters/ 相同
    ├── tool_to_l2.pt      # 与 clusters/ 相同
    └── m_global_tool.pt   # 与 m_global.pt 完全相同（但改名为强调工具级）

使用方法：
  python scripts/prepare_flat_clusters.py --task task1
  python scripts/prepare_flat_clusters.py --task task2
  python scripts/prepare_flat_clusters.py --task task3
"""

import argparse
import os
import re
import json
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 工具函数（与 prepare_task_data.py 保持一致）
# ---------------------------------------------------------------------------
def normalize(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '', str(name)).lower()


def extract_tool_anchor(assistant_content: str):
    """从 assistant 内容中提取 <<Tool&&Api>> 锚点。"""
    match = re.match(r'<<(.+?)&&(.+?)>>', assistant_content.strip())
    if not match:
        return None, None
    return match.group(1).strip(), match.group(2).strip()


# ---------------------------------------------------------------------------
# 从 m_global.pt 计算 m_global_tool.pt（基于工具->L2 直接映射）
# ---------------------------------------------------------------------------
def compute_m_global_tool_from_m_global(m_global: torch.Tensor,
                                        tool_to_l2,
                                        num_boxes: int,
                                        smoothing: float = 1e-4):
    """
    利用已有的 m_global（[num_tools, num_boxes]）作为工��级先验。

    逻辑：
      m_global 的行索引是 tool_id，列索引是 l2_box_id。
      与 m_global_tool 完全同构，区别只是语义上的命名。
      直接复制即可。

    smoothing: Laplace 平滑（避免完全未出现的工具产生全零行）
    """
    num_tools = m_global.shape[0]
    m_tool = m_global.clone()

    # Laplace 平滑：给每个工具的每个盒子加一个小的先验
    m_tool = m_tool + smoothing

    # 行归一化（每行和为 1，构成概率分布）
    row_sum = m_tool.sum(dim=1, keepdim=True)
    m_tool = m_tool / row_sum

    return m_tool


# ---------------------------------------------------------------------------
# 真正从序列数据计算 m_global_tool（备选，如果 m_global.pt 不存在）
# ---------------------------------------------------------------------------
def build_m_global_tool_from_sequences(memorization_json: str,
                                      tool_to_l2,
                                      num_tools: int,
                                      num_boxes: int,
                                      tools_with_id_json: str = None,
                                      smoothing: float = 1e-4):
    """
    从 memorization_train.json 的多轮对话序列中，
    统计 tool -> next_l2_box 的转移频率，构建 m_global_tool。

    格式：[num_tools, num_boxes]
    m_global_tool[t, l2] = P(next_l2=l2 | current_tool=t)
    """
    with open(memorization_json, encoding="utf-8") as f:
        memo_data = json.load(f)

    # 构建 anchor -> tool_id 映射（如果提供了 tools_with_id）
    anchor_to_id = {}
    if tools_with_id_json and os.path.exists(tools_with_id_json):
        with open(tools_with_id_json, encoding="utf-8") as f:
            tools_with_id = json.load(f)
        for tid_str, text in tools_with_id.items():
            m_t = re.search(r'Tool:\s*(.*?)\.', text)
            m_a = re.search(r'API:\s*(.*?)\.', text)
            if m_t and m_a:
                t_name = m_t.group(1).strip()
                a_name = m_a.group(1).strip()
                norm = normalize(f"{a_name}for{t_name}")
                anchor_to_id[norm] = int(tid_str)

    # 统计计数矩阵：[num_tools, num_boxes]
    counts = torch.zeros((num_tools, num_boxes), dtype=torch.float32)

    for item in tqdm(memo_data, desc="统计工具转移"):
        convs = item.get("conversations", [])
        seq = []
        for conv in convs:
            if conv.get("role") == "assistant":
                t_name, a_name = extract_tool_anchor(conv.get("content", ""))
                if t_name and a_name:
                    norm = normalize(f"{a_name}for{t_name}")
                    tool_id = anchor_to_id.get(norm)
                    if tool_id is None:
                        continue
                    seq.append(tool_id)

        # 多轮序列中，统计 t[i] -> next_l2 的转移
        for i in range(len(seq) - 1):
            curr_tool = seq[i]
            # curr_tool 的下一个工具是 seq[i+1]
            # 求 seq[i+1] 对应的 l2_box_id
            next_tool = seq[i + 1]
            if next_tool not in tool_to_l2:
                continue
            next_l2 = tool_to_l2[next_tool]
            if next_l2 < num_boxes:
                counts[curr_tool, next_l2] += 1

    # Laplace 平滑 + 行归一化
    counts = counts + smoothing
    row_sum = counts.sum(dim=1, keepdim=True)
    m_tool = counts / row_sum

    return m_tool


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------
TASK_CONFIGS = {
    "task1": {
        "cluster_dir": "./data/task1/clusters",
        "flat_cluster_dir": "./data/task1/flat_clusters",
        "raw_dir": "./data/task1/raw",
        "start_tool_id": 11112,
    },
    "task2": {
        "cluster_dir": "./data/task2/clusters",
        "flat_cluster_dir": "./data/task2/flat_clusters",
        "raw_dir": "./data/task2/raw",
        "start_tool_id": 11752,
    },
    "task3": {
        "cluster_dir": "./data/task3/clusters",
        "flat_cluster_dir": "./data/task3/flat_clusters",
        "raw_dir": "./data/task3/raw",
        "start_tool_id": 12392,
    },
    "base": {
        "cluster_dir": "./data/train/clusters",
        "flat_cluster_dir": "./data/train/flat_clusters",
        "raw_dir": "./data/train/raw",
        "start_tool_id": 0,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="生成扁平消融实验所需的 flat_clusters 目录"
    )
    parser.add_argument(
        "--task", required=True,
        choices=["base", "task1", "task2", "task3"],
        help="目标任务"
    )
    parser.add_argument(
        "--method", default="copy_from_m_global",
        choices=["copy_from_m_global", "build_from_sequences"],
        help=(
            "生成方式：copy_from_m_global（默认，直接复制 m_global.pt）"
            "或 build_from_sequences（从 memorization 序列统计）"
        )
    )
    parser.add_argument(
        "--smoothing", type=float, default=1e-4,
        help="Laplace 平滑系数（默认 1e-4）"
    )
    args = parser.parse_args()

    if args.task not in TASK_CONFIGS:
        raise ValueError(f"未知任务: {args.task}")

    cfg = TASK_CONFIGS[args.task]
    cluster_dir = cfg["cluster_dir"]
    flat_dir = cfg["flat_cluster_dir"]
    raw_dir = cfg["raw_dir"]

    os.makedirs(flat_dir, exist_ok=True)

    # --- 加载原始聚类数据 ---
    l2_centers = torch.load(os.path.join(cluster_dir, "l2_centers.pt"), weights_only=False)
    l2_widths  = torch.load(os.path.join(cluster_dir, "l2_widths.pt"),  weights_only=False)
    tool_to_l2 = torch.load(os.path.join(cluster_dir, "tool_to_l2.pt"), weights_only=False)
    m_global   = torch.load(os.path.join(cluster_dir, "m_global.pt"),   weights_only=False)

    num_tools = m_global.shape[0]
    num_boxes = l2_centers.shape[0]
    dim = l2_centers.shape[1]

    print(f"\n{'='*50}")
    print(f"[扁平空间生成] 任务={args.task}  方法={args.method}")
    print(f"{'='*50}")
    print(f"  工具数: {num_tools}，L2 盒子数: {num_boxes}，维度: {dim}")

    # --- 计算 m_global_tool ---
    if args.method == "copy_from_m_global":
        # 核心：m_global.pt 的 shape 已经是 [num_tools, num_boxes]
        # 与 m_global_tool 完全同构，直接复制即可
        m_global_tool = compute_m_global_tool_from_m_global(
            m_global, tool_to_l2, num_boxes, smoothing=args.smoothing
        )
        print(f"  -> 从 m_global.pt 复制（已应用 Laplace 平滑）")
    else:
        memo_path = os.path.join(raw_dir, "memorization_train.json")
        tools_json = os.path.join(cluster_dir, f"{args.task}_tools_with_id.json")
        m_global_tool = build_m_global_tool_from_sequences(
            memo_path, tool_to_l2, num_tools, num_boxes,
            tools_with_id_json=tools_json,
            smoothing=args.smoothing
        )
        print(f"  -> 从 memorization 序列统计（已应用 Laplace 平滑）")

    # --- 保存到 flat_clusters/ ---
    torch.save(l2_centers,  os.path.join(flat_dir, "l2_centers.pt"))
    torch.save(l2_widths,   os.path.join(flat_dir, "l2_widths.pt"))
    torch.save(tool_to_l2,   os.path.join(flat_dir, "tool_to_l2.pt"))
    torch.save(m_global_tool, os.path.join(flat_dir, "m_global_tool.pt"))

    print(f"\n  生成文件:")
    for fname in ["l2_centers.pt", "l2_widths.pt", "tool_to_l2.pt", "m_global_tool.pt"]:
        fpath = os.path.join(flat_dir, fname)
        size_mb = os.path.getsize(fpath) / 1024 / 1024
        print(f"    {fname:<25} {size_mb:.2f} MB")

    # --- 验证 ---
    mgt = torch.load(os.path.join(flat_dir, "m_global_tool.pt"), weights_only=False)
    print(f"\n  验证 m_global_tool:")
    print(f"    shape:   {mgt.shape}  （应为 [{num_tools}, {num_boxes}]）")
    print(f"    行和:    {mgt.sum(dim=1)[:5].tolist()}  （应为全 1）")
    print(f"    范围:    [{mgt.min().item():.6f}, {mgt.max().item():.6f}]")
    print(f"\n  保存目录: {flat_dir}/")
    print(f"  ✅ 完成！")


if __name__ == "__main__":
    main()
