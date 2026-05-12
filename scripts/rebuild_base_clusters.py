#!/usr/bin/env python3
"""
scripts/rebuild_base_clusters.py
================================
重跑 Base 任务的聚类流水线（独立于 prepare_task_data.py）。

流程：
  1. 加载 data/train/raw/train_tools_with_id.json（11112 个工具）
  2. 用 SentenceTransformer 提取 384 维特征
  3. L1: 按 l1_domain 分组（50 个），取每组特征均值作为 L1 center
  4. L2: 每个 L1 组内跑 HDBSCAN，生成 L2 boxes
  5. 保存 clusters/ 下所有文件

使用：
  python scripts/rebuild_base_clusters.py
"""
import json, os, torch, numpy as np, argparse
from collections import defaultdict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import hdbscan

ENCODER_MODEL = "all-MiniLM-L6-v2"
EPSILON = 1e-4
OUT_DIR = "./data/train/clusters"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("Base 聚类重建流水线")
    print("=" * 60)

    # ── 1. 加载 base 工具 ──────────────────────────────────────────
    tools_path = "./data/train/raw/train_tools_with_id.json"
    with open(tools_path, encoding="utf-8") as f:
        base_tools = json.load(f)
    print(f"  加载 {len(base_tools)} 个 base 工具")

    # 读取 l1_names.json（如果存在则复用，否则从数据重建）
    l1_names_path = os.path.join(OUT_DIR, "l1_names.json")
    if os.path.exists(l1_names_path):
        with open(l1_names_path, encoding="utf-8") as f:
            l1_name_to_idx = {k: int(v) for k, v in json.load(f).items()}
        print(f"  复用 l1_names.json: {len(l1_name_to_idx)} 个 L1 领域")
    else:
        # 从 base_tools 提取所有 l1_domain
        unique_l1 = sorted(set(t["l1_domain"] for t in base_tools))
        l1_name_to_idx = {name: i for i, name in enumerate(unique_l1)}
        with open(l1_names_path, "w", encoding="utf-8") as f:
            json.dump({str(v): k for k, v in l1_name_to_idx.items()}, f)
        print(f"  新建 l1_names.json: {len(l1_name_to_idx)} 个 L1 领域")

    # 工具文本列表（按原始顺序）
    tool_texts = []
    tool_l1_names = []
    tool_ids = []
    for i, t in enumerate(base_tools):
        l1 = t["l1_domain"]
        # 拼接文本用于 embedding
        text = (f"Tool: {t['tool_name']}. "
                f"Description: {t.get('api_description', '')}. "
                f"API: {t['api_name']}.")
        tool_texts.append(text)
        tool_l1_names.append(l1)
        tool_ids.append(i)

    num_tools = len(tool_texts)
    num_l1 = len(l1_name_to_idx)
    print(f"  工具数: {num_tools}，L1 领域数: {num_l1}")

    # ── 2. 特征提取 ────────────────────────────────────────────────
    print("  加载 SentenceTransformer，提取 384 维特征...")
    encoder = SentenceTransformer(ENCODER_MODEL, device="cpu")
    vectors = encoder.encode(tool_texts, batch_size=256, convert_to_tensor=True,
                            show_progress_bar=True).cpu().float()
    vectors_np = vectors.numpy()
    print(f"  特征维度: {vectors.shape}")

    # ── 3. 建立 L1 centers ─────────────────────────────────────────
    l1_centers = torch.zeros((num_l1, vectors.shape[1]), dtype=torch.float32)
    l1_groups = defaultdict(list)
    for i, l1_name in enumerate(tool_l1_names):
        l1_groups[l1_name].append(i)
    for l1_name, indices in l1_groups.items():
        l1_idx = l1_name_to_idx[l1_name]
        group_vecs = vectors[indices]
        l1_centers[l1_idx] = group_vecs.mean(dim=0)
    l1_widths = torch.ones_like(l1_centers) * 0.5

    print(f"  L1 centers: {l1_centers.shape}")
    for l1_name, indices in sorted(l1_groups.items(), key=lambda x: -len(x[1])):
        print(f"    {l1_name}: {len(indices)} 个工具")

    # ── 4. 每个 L1 组内跑 HDBSCAN，生成 L2 boxes ──────────────────
    new_tool_to_l2 = {}
    new_l2_to_l1 = {}
    l2_boxes = []  # list of (center, width)
    next_l2_id = 0

    for l1_name, group_indices in tqdm(l1_groups.items(), desc="HDBSCAN per L1"):
        l1_id = l1_name_to_idx[l1_name]
        group_vecs_np = vectors_np[group_indices]

        if len(group_indices) >= 3:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric='euclidean')
            labels = clusterer.fit_predict(group_vecs_np)

            # 第一遍：只为有效簇建盒子
            local_to_global = {}
            for local_idx, label in enumerate(labels):
                if label != -1 and label not in local_to_global:
                    gid = next_l2_id
                    local_to_global[label] = gid
                    next_l2_id += 1
                    new_l2_to_l1[gid] = l1_id
                    mask = (labels == label)
                    member_vecs = vectors[group_indices][mask]
                    center = (torch.max(member_vecs, dim=0)[0] + torch.min(member_vecs, dim=0)[0]) / 2.0
                    width  = (torch.max(member_vecs, dim=0)[0] - torch.min(member_vecs, dim=0)[0]) / 2.0 + EPSILON
                    l2_boxes.append((center, width))

            # 第二遍：分配所有工具（有效簇 + 噪声点）
            all_centers = torch.stack([b[0] for b in l2_boxes]) if l2_boxes else None
            for local_idx, label in enumerate(labels):
                global_tool_id = group_indices[local_idx]
                if label != -1:
                    gid = local_to_global[label]
                else:
                    # 噪声点：分配到最近的已有盒子
                    if all_centers is not None and all_centers.shape[0] > 0:
                        gid = closest_l2_for_vector(vectors[global_tool_id], all_centers)
                    else:
                        continue  # 当前 L1 组内无任何盒子
                new_tool_to_l2[global_tool_id] = gid
        else:
            # < 3 个工具：直接用 1 个 L2 box 包裹
            gid = next_l2_id
            next_l2_id += 1
            group_vecs_t = vectors[group_indices]
            center = (torch.max(group_vecs_t, dim=0)[0] + torch.min(group_vecs_t, dim=0)[0]) / 2.0
            width  = (torch.max(group_vecs_t, dim=0)[0] - torch.min(group_vecs_t, dim=0)[0]) / 2.0 + EPSILON
            l2_boxes.append((center, width))
            new_l2_to_l1[gid] = l1_id
            for gi in group_indices:
                new_tool_to_l2[gi] = gid

    l2_centers = torch.stack([b[0] for b in l2_boxes])
    l2_widths  = torch.stack([b[1] for b in l2_boxes])
    print(f"  L2 boxes: {l2_centers.shape[0]} 个")

    # ── 5. 构建 m_global ───────────────────────────────────────────
    total_boxes = l2_centers.shape[0]
    m_global = torch.zeros((num_tools, total_boxes), dtype=torch.float32)
    for tool_id, l2_id in new_tool_to_l2.items():
        m_global[tool_id, l2_id] = 1.0
    # 行归一化
    row_sums = m_global.sum(dim=1, keepdim=True).clamp(min=1e-9)
    m_global = m_global / row_sums

    # ── 6. 存盘 ────────────────────────────────────────────────────
    torch.save(l1_centers, os.path.join(OUT_DIR, "l1_centers.pt"))
    torch.save(l1_widths,  os.path.join(OUT_DIR, "l1_widths.pt"))
    torch.save(l2_centers, os.path.join(OUT_DIR, "l2_centers.pt"))
    torch.save(l2_widths,  os.path.join(OUT_DIR, "l2_widths.pt"))
    torch.save(new_tool_to_l2, os.path.join(OUT_DIR, "tool_to_l2.pt"))
    torch.save(new_l2_to_l1,   os.path.join(OUT_DIR, "l2_to_l1.pt"))
    torch.save(m_global,        os.path.join(OUT_DIR, "m_global.pt"))

    print(f"\n  ✅ 聚类完成！")
    print(f"     L1: {num_l1} 个，维度 {vectors.shape[1]}")
    print(f"     L2: {total_boxes} 个")
    print(f"     工具: {num_tools} 个，m_global: {m_global.shape}")
    print(f"     输出目录: {OUT_DIR}/")


def closest_l2_for_vector(vec, l2_centers):
    """返回距离最近的 L2 box id"""
    if l2_centers.shape[0] == 0:
        return 0
    dists = torch.norm(l2_centers - vec.unsqueeze(0), dim=1)
    return torch.argmin(dists).item()


if __name__ == "__main__":
    main()
