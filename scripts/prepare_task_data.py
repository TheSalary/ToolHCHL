#!/usr/bin/env python3
"""
scripts/prepare_task_data.py
=============================
IH-PromptDSI 通用数据准备脚本。
完整流水线包含三个阶段，全部通过 --stage 参数控制：

  Stage 0  检查数据：统计 raw 文件行数、唯一工具数、题库样本数。
  Stage 1  从 memorization_train.json 提取工具真名，派发全局 ID，
            生成 clusters/{task}_tools_with_id.json。
  Stage 2  用 tools.txt 归一化检索题库（retrieval_train.json），
            将工具名替换为全局 ID，生成 clusters/{task}_retrieval_clean.json。
  Stage 3  特征提取（SentenceTransformer） + 增量 KMeans 聚类
            + 缝合新 L2 盒子 + 扩展 m_global，输出完整 clusters/ 目录。

使用示例：
  # 完整流水线（适合新任务）
  python scripts/prepare_task_data.py --task task2

  # 仅检查数据
  python scripts/prepare_task_data.py --task task2 --stage 0

  # 跳过 stage 1/2，直接重做聚类
  python scripts/prepare_task_data.py --task task2 --stage 3
"""

import argparse
import json
import os
import re
import sys
import torch
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import hdbscan
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# 通用工具函数
# ---------------------------------------------------------------------------
EPSILON = 1e-4
ENCODER_MODEL = "all-MiniLM-L6-v2"


def normalize(name: str) -> str:
    """暴力标准化：剔除所有非字母数字字符，全部转小写。"""
    return re.sub(r'[^a-zA-Z0-9]', '', str(name)).lower()


def extract_tool_anchor(assistant_content: str):
    """从 assistant 内容中提取 <<Tool&&Api>> 锚点。"""
    match = re.match(r'<<(.+?)&&(.+?)>>', assistant_content.strip())
    if not match:
        return None, None
    return match.group(1).strip(), match.group(2).strip()


def build_clean_text(user_content: str, tool_name: str, api_name: str) -> str:
    """从 user content 中解析四个字段，拼接为干净文本。"""
    regex = (
        r"Tool Name:\s*(.*?)\s*"
        r"Tool Description:\s*(.*?)\s*"
        r"Api Name:\s*(.*?)\s*"
        r"Api Description:\s*(.*)"
    )
    m = re.search(regex, user_content, re.DOTALL)
    if m:
        t_name = m.group(1).strip()
        t_desc = m.group(2).strip()
        a_name = m.group(3).strip()
        a_desc = m.group(4).strip()
        return f"Tool: {t_name}. Description: {t_desc}. API: {a_name}. API Description: {a_desc}"
    return f"Tool: {tool_name}. API: {api_name}."


# ---------------------------------------------------------------------------
# Stage 0：数据检查
# ---------------------------------------------------------------------------
def stage0(cfg, prev_cfg):
    task = cfg["task_name"]
    print(f"\n{'='*50}")
    print(f"[Stage 0] 数据完整性检查 — {task}")
    print(f"{'='*50}")

    raw_dir = cfg["raw_dir"]

    # memorization
    memo_path = os.path.join(raw_dir, "memorization_train.json")
    if os.path.exists(memo_path):
        with open(memo_path, encoding="utf-8") as f:
            memo_data = json.load(f)
        # 统计唯一工具
        seen = set()
        for item in memo_data:
            for conv in item.get("conversations", []):
                if conv.get("role") == "assistant":
                    t, a = extract_tool_anchor(conv.get("content", ""))
                    if t and a:
                        seen.add(f"{t}||{a}")
        print(f"  memorization_train.json : {len(memo_data)} 条对话，{len(seen)} 个唯一工具")
    else:
        print(f"  memorization_train.json : 不存在！")

    # retrieval_train
    train_path = os.path.join(raw_dir, "retrieval_train.json")
    if os.path.exists(train_path):
        with open(train_path, encoding="utf-8") as f:
            train_data = json.load(f)
        print(f"  retrieval_train.json    : {len(train_data)} 条")
    else:
        print(f"  retrieval_train.json   : 不存在！")

    # retrieval_eval
    eval_path = os.path.join(raw_dir, "retrieval_eval.json")
    if os.path.exists(eval_path):
        with open(eval_path, encoding="utf-8") as f:
            eval_data = json.load(f)
        print(f"  retrieval_eval.json     : {len(eval_data)} 条")
    else:
        print(f"  retrieval_eval.json     : 不存在！")

    # tools.txt
    tools_txt_path = os.path.join(raw_dir, "tools.txt")
    if os.path.exists(tools_txt_path):
        with open(tools_txt_path, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        print(f"  tools.txt              : {len(lines)} 个工具")
    else:
        print(f"  tools.txt              : 不存在！")

    # 检查继承链
    if prev_cfg:
        prev_cluster_dir = prev_cfg["cluster_dir"]
        prev_l2 = os.path.join(prev_cluster_dir, "l2_centers.pt")
        if os.path.exists(prev_l2):
            prev_l2c = torch.load(prev_l2, weights_only=False)
            print(f"\n  继承自 {cfg['prev_task']} 的 l2_centers: {prev_l2c.shape}")
        else:
            print(f"\n  警告：继承的 l2_centers 不存在: {prev_l2}")
    print()


# ---------------------------------------------------------------------------
# Stage 1：从 memorization 提取工具，生成 *_tools_with_id.json
# ---------------------------------------------------------------------------
def stage1(cfg, prev_cfg):
    task = cfg["task_name"]
    raw_dir = cfg["raw_dir"]
    cluster_dir = cfg["cluster_dir"]
    start_id = cfg["start_tool_id"]

    os.makedirs(cluster_dir, exist_ok=True)
    out_tools_json = os.path.join(cluster_dir, f"{task}_tools_with_id.json")

    memo_path = os.path.join(raw_dir, "memorization_train.json")
    print(f"\n{'='*50}")
    print(f"[Stage 1] 提取工具并派发 ID — {task}")
    print(f"{'='*50}")
    print(f"  输入: {memo_path}")
    print(f"  输出: {out_tools_json}")
    print(f"  起始 ID: {start_id}")

    with open(memo_path, encoding="utf-8") as f:
        memo_data = json.load(f)

    # 建立 base 的 (tool_name -> l1_domain) 查表（base 是 prev_task，路径从 prev_cfg 读取）
    if prev_cfg:
        base_path = os.path.join(prev_cfg["raw_dir"], "train_tools_with_id.json")
    else:
        base_path = os.path.join(raw_dir, "train_tools_with_id.json")
    print(f"  加载 l1_domain 查表: {base_path}")
    from collections import defaultdict
    tool_to_l1 = defaultdict(set)
    with open(base_path, encoding="utf-8") as f:
        for t in json.load(f):
            tool_to_l1[t["tool_name"].strip()].add(t["l1_domain"])

    seen = set()
    tools_with_id = {}   # {id: {"l1_domain": str, "text": str}}
    id_from_anchor = {}
    current_id = start_id

    for item in tqdm(memo_data, desc="解析 memo"):
        user_content, assistant_content = "", ""
        for conv in item.get("conversations", []):
            if conv.get("role") == "user":
                user_content = conv.get("content", "")
            elif conv.get("role") == "assistant":
                assistant_content = conv.get("content", "").strip()

        tool_name, api_name = extract_tool_anchor(assistant_content)
        if not tool_name or not api_name:
            continue

        anchor = f"<<{tool_name}&&{api_name}>>"
        if anchor in seen:
            continue
        seen.add(anchor)

        # 用 base 的 tool_name 匹配 l1_domain
        l1_domains = tool_to_l1.get(tool_name.strip(), set())
        if len(l1_domains) == 1:
            l1_domain = next(iter(l1_domains))
        elif len(l1_domains) > 1:
            l1_domain = next(iter(l1_domains))  # 取第一个
        else:
            l1_domain = "Unknown"

        clean_text = build_clean_text(user_content, tool_name, api_name)
        tools_with_id[str(current_id)] = {"l1_domain": l1_domain, "text": clean_text}
        id_from_anchor[anchor] = current_id
        current_id += 1

    with open(out_tools_json, "w", encoding="utf-8") as f:
        json.dump(tools_with_id, f, ensure_ascii=False, indent=2)

    num_new_tools = len(tools_with_id)
    unknown_cnt = sum(1 for v in tools_with_id.values() if v["l1_domain"] == "Unknown")
    print(f"  -> 提取完毕！共 {num_new_tools} 个新工具，ID: {start_id} ~ {current_id - 1}")
    print(f"     已知 l1_domain: {num_new_tools - unknown_cnt}，未知 (embedding 估算): {unknown_cnt}")

    # 兼容旧写法：同时复制一份到 raw/ 目录（供 eval_taskN.py 直接读取）
    raw_tools_json = os.path.join(raw_dir, f"{task}_tools_with_id.json")
    if not os.path.exists(raw_tools_json):
        with open(raw_tools_json, "w", encoding="utf-8") as f:
            json.dump(tools_with_id, f, ensure_ascii=False, indent=2)
        print(f"  -> 同时写入 raw/ 兼容路径: {raw_tools_json}")

    # stage1 需要 l1_names.json（stage3 依赖它）
    l1_names_src = os.path.join(prev_cfg["cluster_dir"], "l1_names.json") if prev_cfg else None
    if l1_names_src and os.path.exists(l1_names_src):
        l1_names_dst = os.path.join(cluster_dir, "l1_names.json")
        if not os.path.exists(l1_names_dst):
            with open(l1_names_src, encoding="utf-8") as src, \
                 open(l1_names_dst, "w", encoding="utf-8") as dst:
                dst.write(src.read())
            print(f"  -> 复制 l1_names.json → {l1_names_dst}")

    return tools_with_id, id_from_anchor, num_new_tools


# ---------------------------------------------------------------------------
# Stage 2：用 tools.txt 归一化题库，生成 *_retrieval_clean.json
# ---------------------------------------------------------------------------
def stage2(cfg, prev_cfg):
    task = cfg["task_name"]
    raw_dir = cfg["raw_dir"]
    cluster_dir = cfg["cluster_dir"]

    # 如果 stage 1 未运行，先尝试加载 tools_with_id
    tools_json_path = os.path.join(cluster_dir, f"{task}_tools_with_id.json")
    if not os.path.exists(tools_json_path):
        raw_tools_json = os.path.join(raw_dir, f"{task}_tools_with_id.json")
        if os.path.exists(raw_tools_json):
            with open(raw_tools_json, encoding="utf-8") as f:
                tools_with_id = json.load(f)
        else:
            raise FileNotFoundError(
                f"找不到 tools_with_id.json，请先运行 stage 1：\n"
                f"  {tools_json_path}\n"
                f"  或\n"
                f"  {raw_tools_json}"
            )
    else:
        with open(tools_json_path, encoding="utf-8") as f:
            tools_with_id = json.load(f)

    # 从 tools_with_id 反建 name -> id 映射（用于 stage 2 的正则提取）
    # 这个映射对应的是 tools.txt 中的行，格式为 "<<Tool&&Api>>"
    name_to_id = {}
    for tid_str, text in tools_with_id.items():
        m_t = re.search(r'Tool:\s*(.*?)\.', text)
        m_a = re.search(r'API:\s*(.*?)\.', text)
        if m_t and m_a:
            t_name = m_t.group(1).strip()
            a_name = m_a.group(1).strip()
            norm = normalize(f"{a_name}for{t_name}")
            name_to_id[norm] = int(tid_str)

    # 加载 tools.txt 归一化，建立 <<Tool&&Api>> -> id
    tools_txt_path = os.path.join(raw_dir, "tools.txt")
    if not os.path.exists(tools_txt_path):
        raise FileNotFoundError(f"找不到 tools.txt: {tools_txt_path}")

    with open(tools_txt_path, encoding="utf-8") as f:
        txt_lines = [l.strip() for l in f if l.strip()]

    for line in txt_lines:
        m = re.match(r'<<(.+?)&&(.+?)>>', line)
        if m:
            t, a = m.group(1).strip(), m.group(2).strip()
            norm = normalize(f"{a}for{t}")
            if norm in name_to_id:
                continue  # 已在 tools_with_id 中
            # 如果 tools.txt 中有但 memo 中没有，用临时占位（跳过）

    print(f"\n{'='*50}")
    print(f"[Stage 2] 归一化检索题库 — {task}")
    print(f"{'='*50}")

    retrieval_path = os.path.join(raw_dir, "retrieval_train.json")
    out_clean = os.path.join(cluster_dir, f"{task}_retrieval_clean.json")

    with open(retrieval_path, encoding="utf-8") as f:
        ret_data = json.load(f)

    clean_samples = []
    hit, miss = 0, 0

    for item in tqdm(ret_data, desc="解析 retrieval"):
        query, target_str = "", ""
        for conv in item.get("conversations", []):
            if conv.get("role") == "user":
                query = conv.get("content", "").strip()
            elif conv.get("role") == "assistant":
                target_str = conv.get("content", "").strip()

        if not query or not target_str:
            continue

        m_target = re.match(r'<<(.+?)&&(.+?)>>', target_str)
        if not m_target:
            miss += 1
            continue

        t_name = m_target.group(1).strip()
        a_name = m_target.group(2).strip()
        norm = normalize(f"{a_name}for{t_name}")

        if norm in name_to_id:
            target_id = name_to_id[norm]
            clean_samples.append({
                "query": query,
                "target_id": target_id,
                "target_name": target_str
            })
            hit += 1
        else:
            miss += 1

    with open(out_clean, "w", encoding="utf-8") as f:
        json.dump(clean_samples, f, ensure_ascii=False, indent=2)

    print(f"  -> 归一化完毕！命中 {hit} 条，跳过 {miss} 条")
    print(f"  -> 保存至: {out_clean}")
    return clean_samples


# ---------------------------------------------------------------------------
# Stage 3：特征提取 + 增量聚类 + 缝合 L2 盒子 + 扩展 m_global
# ---------------------------------------------------------------------------
def stage3(cfg, prev_cfg):
    task = cfg["task_name"]
    cluster_dir = cfg["cluster_dir"]
    prev_cluster_dir = prev_cfg["cluster_dir"]

    os.makedirs(cluster_dir, exist_ok=True)

    tools_json_path = os.path.join(cluster_dir, f"{task}_tools_with_id.json")
    with open(tools_json_path, encoding="utf-8") as f:
        tools_with_id = json.load(f)

    # 读取 l1_name -> idx 映射
    l1_names_path = os.path.join(prev_cfg["cluster_dir"], "l1_names.json")
    with open(l1_names_path, encoding="utf-8") as f:
        l1_name_to_idx = {k: int(v) for k, v in json.load(f).items()}
    num_l1 = len(l1_name_to_idx)

    # 提取文本和 l1_domain（stage1 已写入）
    tool_ids = sorted(tools_with_id.keys(), key=int)
    tool_texts = [tools_with_id[tid]["text"] for tid in tool_ids]
    tool_l1s   = [tools_with_id[tid]["l1_domain"] for tid in tool_ids]
    num_new_tools = len(tool_ids)

    print(f"\n{'='*50}")
    print(f"[Stage 3] HDBSCAN 增量聚类 + 空间缝合 — {task}")
    print(f"{'='*50}")
    print(f"  新工具数: {num_new_tools}，L1 固定 {num_l1} 个")
    print(f"  继承自: {prev_cluster_dir}")

    # --- 加载旧物理空间 ---
    old_l1_centers = torch.load(
        prev_cfg["l1_centers"], weights_only=False
    ).float()
    old_l1_widths = torch.load(
        prev_cfg["l1_widths"], weights_only=False
    ).float()
    old_l2_centers = torch.load(
        os.path.join(prev_cluster_dir, "l2_centers.pt"), weights_only=False
    ).float()
    old_l2_widths = torch.load(
        os.path.join(prev_cluster_dir, "l2_widths.pt"), weights_only=False
    ).float()
    old_tool_to_l2 = torch.load(
        os.path.join(prev_cluster_dir, "tool_to_l2.pt"), weights_only=False
    )
    old_l2_to_l1 = torch.load(
        os.path.join(prev_cluster_dir, "l2_to_l1.pt"), weights_only=False
    )
    old_m_global = torch.load(
        os.path.join(prev_cluster_dir, "m_global.pt"), weights_only=False
    )

    old_num_boxes = old_l2_centers.shape[0]
    dim = old_l2_centers.shape[1]
    print(f"  旧空间: {num_l1} 个 L1，{old_num_boxes} 个 L2 盒子，维度 {dim}")

    # --- 特征提取（仅对需要聚类的工具）---
    print("  -> 加载 SentenceTransformer，提取 384 维特征...")
    encoder = SentenceTransformer(ENCODER_MODEL, device="cpu")
    vectors = encoder.encode(tool_texts, convert_to_tensor=True,
                             show_progress_bar=True).cpu().float()
    vectors_np = vectors.numpy()

    # --- 增量 HDBSCAN 聚类 ---
    # 步骤 1: 按 l1_domain 分组（已知）；未知领域用 embedding 距离找最近 L1
    l1_groups = {i: [] for i in range(num_l1)}
    unknown_indices = []   # l1_domain == "Unknown" 的索引

    for idx in range(num_new_tools):
        l1_name = tool_l1s[idx]
        if l1_name in l1_name_to_idx:
            l1_groups[l1_name_to_idx[l1_name]].append(idx)
        else:
            # 未知领域：用 embedding 距离找最近 L1（与 base 一致）
            dist_to_l1 = torch.norm(old_l1_centers - vectors[idx].unsqueeze(0), dim=1)
            closest_l1 = torch.argmin(dist_to_l1).item()
            l1_groups[closest_l1].append(idx)
            unknown_indices.append(idx)

    known_domains = num_new_tools - len(unknown_indices)
    print(f"  -> 按 l1_domain 分组完成：已知领域 {known_domains} 个，embedding 估算 {len(unknown_indices)} 个")
    if unknown_indices:
        print(f"     未知领域工具数: {len(unknown_indices)}")

    # 步骤 2: 每个 L1 组内跑 HDBSCAN
    new_l2_boxes = []
    new_tool_to_l2 = dict(old_tool_to_l2)
    new_l2_to_l1 = dict(old_l2_to_l1)
    next_box_id = old_num_boxes
    start_tool_id = cfg["start_tool_id"]

    for l1_id, group_indices in tqdm(l1_groups.items(), desc="HDBSCAN per L1"):
        if not group_indices:
            continue
        group_vecs = vectors_np[group_indices]

        if len(group_indices) >= 3:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric='euclidean')
            labels = clusterer.fit_predict(group_vecs)

            local_to_global = {}
            for local_idx, label in enumerate(labels):
                global_tool_id = start_tool_id + group_indices[local_idx]
                if label == -1:
                    # 噪声点：分配到最近盒子
                    gid = closest_l2_for_vector(
                        vectors[group_indices[local_idx]], old_l2_centers
                    )
                    new_tool_to_l2[global_tool_id] = gid
                else:
                    if label not in local_to_global:
                        gid = next_box_id
                        local_to_global[label] = gid
                        next_box_id += 1
                        new_l2_to_l1[gid] = l1_id
                        # 建立新盒子
                        mask = (labels == label)
                        member_vecs = vectors[group_indices][mask]
                        center = (torch.max(member_vecs, dim=0)[0] + torch.min(member_vecs, dim=0)[0]) / 2.0
                        width  = (torch.max(member_vecs, dim=0)[0] - torch.min(member_vecs, dim=0)[0]) / 2.0 + EPSILON
                        new_l2_boxes.append((center, width))
                    else:
                        gid = local_to_global[label]
                    new_tool_to_l2[global_tool_id] = gid
        else:
            # 工具数 < 3：直接用 1 个 L2 box 包裹
            gid = next_box_id
            next_box_id += 1
            group_vecs_t = vectors[group_indices]
            center = (torch.max(group_vecs_t, dim=0)[0] + torch.min(group_vecs_t, dim=0)[0]) / 2.0
            width  = (torch.max(group_vecs_t, dim=0)[0] - torch.min(group_vecs_t, dim=0)[0]) / 2.0 + EPSILON
            new_l2_boxes.append((center, width))
            new_l2_to_l1[gid] = l1_id
            for gi in group_indices:
                new_tool_to_l2[start_tool_id + gi] = gid

    valid_new_boxes = len(new_l2_boxes)
    print(f"  -> HDBSCAN 完成！新增 {valid_new_boxes} 个 L2 盒子")

    # --- 组装新的 L2 空间 ---
    new_centers = torch.stack([b[0] for b in new_l2_boxes])
    new_widths  = torch.stack([b[1] for b in new_l2_boxes])
    combined_l2_centers = torch.cat([old_l2_centers, new_centers], dim=0)
    combined_l2_widths   = torch.cat([old_l2_widths,   new_widths],  dim=0) \
        if "old_l2_widths" in dir() else torch.cat([old_l2_centers, new_widths], dim=0)

    # --- 扩展 m_global ---
    total_tools = old_m_global.shape[0] + num_new_tools
    total_boxes = old_num_boxes + valid_new_boxes
    new_m = torch.zeros((total_tools, total_boxes), dtype=old_m_global.dtype)
    new_m[:old_m_global.shape[0], :old_m_global.shape[1]] = old_m_global
    new_m[old_m_global.shape[0]:, :] = 1.0 / total_boxes

    # --- 存盘 ---
    torch.save(combined_l2_centers, os.path.join(cluster_dir, "l2_centers.pt"))
    torch.save(combined_l2_widths,  os.path.join(cluster_dir, "l2_widths.pt"))
    torch.save(old_l1_centers,      os.path.join(cluster_dir, "l1_centers.pt"))
    torch.save(old_l1_widths,       os.path.join(cluster_dir, "l1_widths.pt"))
    torch.save(new_tool_to_l2,      os.path.join(cluster_dir, "tool_to_l2.pt"))
    torch.save(new_l2_to_l1,        os.path.join(cluster_dir, "l2_to_l1.pt"))
    torch.save(new_m,               os.path.join(cluster_dir, "m_global.pt"))

    # 兼容旧路径：复制到 raw/
    raw_dir = cfg["raw_dir"]
    for fname in ["l2_centers.pt", "l2_widths.pt", "l1_centers.pt",
                  "l1_widths.pt", "tool_to_l2.pt", "m_global.pt"]:
        src = os.path.join(cluster_dir, fname)
        dst = os.path.join(raw_dir, fname)
        if not os.path.exists(dst):
            import shutil
            shutil.copy(src, dst)

    print(f"  -> 缝合完成！")
    print(f"     L2 盒子总数: {old_num_boxes} -> {total_boxes}")
    print(f"     工具总数:    {old_m_global.shape[0]} -> {total_tools}")
    print(f"     m_global:    {old_m_global.shape} -> {new_m.shape}")
    print(f"  -> 所有文件已保存至: {cluster_dir}/")


def closest_l2_for_vector(vec, l2_centers):
    """将一个向量分配到最近的 L2 盒子（欧氏距离）。"""
    dists = torch.norm(l2_centers - vec.unsqueeze(0), dim=1)
    return torch.argmin(dists).item()


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def build_task_config(task_name: str):
    """
    根据 config.yaml 中的 task 名称构建完整路径配置。
    prev_task 链通过查表手动拼接（避免引入 yaml 解析依赖）。
    """
    # 手动路径表（与 config.yaml 严格对应）
    TASK_CONFIGS = {
        "base": {
            "task_name": "base",
            "prev_task": None,
            "cluster_dir": "./data/train/clusters",
            "raw_dir": "./data/train/raw",
            "start_tool_id": 0,
            "new_num_boxes": 0,
            "l2_centers": "./data/train/clusters/l2_centers.pt",
            "l2_widths": "./data/train/clusters/l2_widths.pt",
            "m_global": "./data/train/clusters/m_global.pt",
            "tool_to_l2": "./data/train/clusters/tool_to_l2.pt",
            "l2_to_l1": "./data/train/clusters/l2_to_l1.pt",
            "l1_centers": "./data/train/clusters/l1_centers.pt",
            "l1_widths": "./data/train/clusters/l1_widths.pt",
        },
        "task1": {
            "task_name": "task1",
            "prev_task": "base",
            "cluster_dir": "./data/task1/clusters",
            "raw_dir": "./data/task1/raw",
            "start_tool_id": 11112,
            "new_num_boxes": 50,
            "l2_centers": "./data/task1/clusters/l2_centers.pt",
            "l2_widths": "./data/task1/clusters/l2_widths.pt",
            "l1_centers": "./data/train/clusters/l1_centers.pt",
            "l1_widths": "./data/train/clusters/l1_widths.pt",
            "m_global": "./data/task1/clusters/m_global.pt",
            "tool_to_l2": "./data/task1/clusters/tool_to_l2.pt",
            "l2_to_l1": "./data/task1/clusters/l2_to_l1.pt",
        },
        "task2": {
            "task_name": "task2",
            "prev_task": "task1",
            "cluster_dir": "./data/task2/clusters",
            "raw_dir": "./data/task2/raw",
            "start_tool_id": 11752,
            "new_num_boxes": 50,
            "l2_centers": "./data/task2/clusters/l2_centers.pt",
            "l2_widths": "./data/task2/clusters/l2_widths.pt",
            "l1_centers": "./data/task1/clusters/l1_centers.pt",
            "l1_widths": "./data/task1/clusters/l1_widths.pt",
            "m_global": "./data/task2/clusters/m_global.pt",
            "tool_to_l2": "./data/task2/clusters/tool_to_l2.pt",
            "l2_to_l1": "./data/task2/clusters/l2_to_l1.pt",
        },
        "task3": {
            "task_name": "task3",
            "prev_task": "task2",
            "cluster_dir": "./data/task3/clusters",
            "raw_dir": "./data/task3/raw",
            "start_tool_id": 12392,
            "new_num_boxes": 50,
            "l2_centers": "./data/task3/clusters/l2_centers.pt",
            "l2_widths": "./data/task3/clusters/l2_widths.pt",
            "l1_centers": "./data/task2/clusters/l1_centers.pt",
            "l1_widths": "./data/task2/clusters/l1_widths.pt",
            "m_global": "./data/task3/clusters/m_global.pt",
            "tool_to_l2": "./data/task3/clusters/tool_to_l2.pt",
            "l2_to_l1": "./data/task3/clusters/l2_to_l1.pt",
        },
    }
    if task_name not in TASK_CONFIGS:
        raise ValueError(f"未知任务: {task_name}，可选: {list(TASK_CONFIGS.keys())}")
    cfg = TASK_CONFIGS[task_name]
    prev_cfg = TASK_CONFIGS[cfg["prev_task"]] if cfg["prev_task"] else None
    return cfg, prev_cfg


def main():
    parser = argparse.ArgumentParser(
        description="IH-PromptDSI 数据准备脚本（支持 stage 0~3）"
    )
    parser.add_argument(
        "--task", required=True,
        choices=["base", "task1", "task2", "task3"],
        help="目标任务名称"
    )
    parser.add_argument(
        "--stage", type=int, default=0,
        choices=[0, 1, 2, 3],
        help="执行阶段：0=检查，1=提取工具，2=归一化题库，3=聚类（默认 0）"
    )
    args = parser.parse_args()

    cfg, prev_cfg = build_task_config(args.task)

    if args.stage == 0:
        stage0(cfg, prev_cfg)
    elif args.stage == 1:
        stage1(cfg, prev_cfg)
    elif args.stage == 2:
        stage2(cfg, prev_cfg)
    elif args.stage == 3:
        if prev_cfg is None:
            raise ValueError("base 任务不需要 stage 3（无继承空间）")
        stage3(cfg, prev_cfg)
        # stage 3 完成后自动检查一下
        print(f"\n{'='*50}")
        print(f"[完成] {args.task} 聚类数据已准备完毕！")
        stage0(cfg, prev_cfg)


if __name__ == "__main__":
    main()
