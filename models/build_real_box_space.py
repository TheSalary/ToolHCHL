import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.cluster import KMeans
import os

# --- 路径配置 ---
input_json_path = "./data/train/raw/train_tools_with_id.json"
output_dir = "./data/train/clusters"

def build_hierarchical_box_space():
    print("1. 正在加载训练集数据...")
    with open(input_json_path, "r", encoding="utf-8") as f:
        tools = json.load(f)
    
    # 2. 向量化
    print("2. 正在进行向量化 (SentenceTransformer)...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu") 
    descriptions = [t.get("api_description", "") for t in tools]
    vectors = encoder.encode(descriptions, show_progress_bar=True)
    
    for i, t in enumerate(tools):
        t["vector"] = vectors[i]

    # 3. 按 L1 领域分组
    l1_groups = {}
    for t in tools:
        l1 = t["l1_domain"]
        if l1 not in l1_groups:
            l1_groups[l1] = []
        l1_groups[l1].append(t)

    # --- 准备存储空间 ---
    dim = 384
    epsilon = 1e-4
    
    l1_names = list(l1_groups.keys())
    num_l1 = len(l1_names)
    l1_centers = torch.zeros(num_l1, dim)
    l1_widths = torch.zeros(num_l1, dim)
    l1_name_to_id = {name: i for i, name in enumerate(l1_names)}

    global_l2_counter = 0
    l2_box_vectors = {} 
    l2_to_l1_map = {}   # 记录每个 L2 属于哪个 L1
    tool_to_l2 = {}

    print(f"3. 正在构建 {num_l1} 个 L1 物理墙并发现 L2 簇...")

    for l1_name, group_tools in l1_groups.items():
        l1_id = l1_name_to_id[l1_name]
        group_vectors = np.array([t["vector"] for t in group_tools])
        group_tensor = torch.tensor(group_vectors, dtype=torch.float32)

        # --- 计算 L1 Box 参数 (整个领域的最大包围盒) ---
        max_l1 = torch.max(group_tensor, dim=0)[0]
        min_l1 = torch.min(group_tensor, dim=0)[0]
        l1_centers[l1_id] = (max_l1 + min_l1) / 2.0
        l1_widths[l1_id] = (max_l1 - min_l1) / 2.0 + epsilon

        # --- 内部 L2 发现 (HDBSCAN)，工具数不足 3 时直接用 1 个 box ---
        if len(group_tools) >= 3:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric='euclidean')
            labels = clusterer.fit_predict(group_vectors)

            local_to_global = {}
            noise_indices = []

            for idx, label in enumerate(labels):
                if label == -1:
                    noise_indices.append(idx)
                else:
                    if label not in local_to_global:
                        local_to_global[label] = global_l2_counter
                        l2_box_vectors[global_l2_counter] = []
                        l2_to_l1_map[global_l2_counter] = l1_id
                        global_l2_counter += 1

                    gid = local_to_global[label]
                    l2_box_vectors[gid].append(group_tools[idx]["vector"])
                    tool_to_l2[group_tools[idx]["tool_id"]] = gid

            # --- HDBSCAN 噪声点：KMeans fallback，确保每个工具都有 L2 归属 ---
            if noise_indices:
                noise_vectors = group_vectors[noise_indices]
                n_noise = len(noise_indices)
                n_clusters = max(1, min(n_noise, 3))
                km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                noise_labels = km.fit_predict(noise_vectors)

                for li, label in enumerate(noise_labels):
                    gid = global_l2_counter + label
                    if gid >= global_l2_counter + n_clusters:
                        continue
                    if gid not in l2_box_vectors:
                        l2_box_vectors[gid] = []
                        l2_to_l1_map[gid] = l1_id
                    l2_box_vectors[gid].append(group_tools[noise_indices[li]]["vector"])
                    tool_to_l2[group_tools[noise_indices[li]]["tool_id"]] = gid
                global_l2_counter += n_clusters

        else:
            # --- 工具数 < 3，直接用 1 个 L2 box 包裹 ---
            for idx, t in enumerate(group_tools):
                gid = global_l2_counter
                if gid not in l2_box_vectors:
                    l2_box_vectors[gid] = []
                    l2_to_l1_map[gid] = l1_id
                    global_l2_counter += 1
                l2_box_vectors[gid].append(t["vector"])
                tool_to_l2[t["tool_id"]] = gid

    # 4. 计算 L2 几何参数
    print(f"4. 正在计算 {global_l2_counter} 个 L2 盒子的几何参数...")
    l2_centers = torch.zeros(global_l2_counter, dim)
    l2_widths = torch.zeros(global_l2_counter, dim)

    for gid, vecs in l2_box_vectors.items():
        vecs_tensor = torch.tensor(np.array(vecs), dtype=torch.float32)
        max_v = torch.max(vecs_tensor, dim=0)[0]
        min_v = torch.min(vecs_tensor, dim=0)[0]
        l2_centers[gid] = (max_v + min_v) / 2.0
        l2_widths[gid] = (max_v - min_v) / 2.0 + epsilon

    # 5. 保存层级空间结果
    os.makedirs(output_dir, exist_ok=True)
    torch.save(l1_centers, os.path.join(output_dir, "l1_centers.pt"))
    torch.save(l1_widths,  os.path.join(output_dir, "l1_widths.pt"))
    torch.save(l2_centers,  os.path.join(output_dir, "l2_centers.pt"))
    torch.save(l2_widths,   os.path.join(output_dir, "l2_widths.pt"))
    torch.save(tool_to_l2,   os.path.join(output_dir, "tool_to_l2.pt"))
    torch.save(l2_to_l1_map, os.path.join(output_dir, "l2_to_l1.pt"))

    # m_global：所有工具 × 所有 L2 box 的均匀先验概率
    total_tools = len(tools)
    total_boxes = global_l2_counter
    m_global = torch.full((total_tools, total_boxes), 1.0 / total_boxes, dtype=torch.float32)
    torch.save(m_global, os.path.join(output_dir, "m_global.pt"))

    # 保存 L1 名字映射，方便以后查阅
    with open(os.path.join(output_dir, "l1_names.json"), "w") as f:
        json.dump(l1_name_to_id, f)

    print(f"\n=== 层级空间构建完成 ===")
    print(f"L1 领域数量: {num_l1}")
    print(f"L2 Box 数量: {global_l2_counter}")
    print(f"工具覆盖: {len(tool_to_l2)}/{total_tools} ({(len(tool_to_l2)/total_tools*100):.1f}%)")
    print(f"m_global 形状: {m_global.shape}")
    print(f"所有层级文件已保存至: {output_dir}")

if __name__ == "__main__":
    build_hierarchical_box_space()