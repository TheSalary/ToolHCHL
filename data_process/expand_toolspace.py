import os
import json
import re
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# ==========================================
# 🚨 1. 路径与参数配置区
# ==========================================
# 老空间的记忆路径 (包含 L1 和 L2)
OLD_L1_CENTERS_PATH = "./data/train/clusters/l1_centers.pt"
OLD_L2_CENTERS_PATH = "./data/train/clusters/l2_centers.pt"
OLD_L2_WIDTHS_PATH = "./data/train/clusters/l2_widths.pt"
OLD_TOOL_TO_L2_PATH = "./data/train/clusters/tool_to_l2.pt"
OLD_L2_TO_L1_PATH = "./data/train/clusters/l2_to_l1.pt"  # 👈 必须加载旧的层级映射

# 新任务（Task 1）的数据路径
TASK1_RAW_TOOLS_JSON = "./data/task1/raw/memorization_train.json" 

# 输出路径（物理隔离保存）
OUT_DIR = "./data/task1/clusters"
OUT_NEW_CENTERS = os.path.join(OUT_DIR, "l2_centers.pt")
OUT_NEW_WIDTHS = os.path.join(OUT_DIR, "l2_widths.pt")
OUT_NEW_TOOL_TO_L2 = os.path.join(OUT_DIR, "tool_to_l2.pt")
OUT_NEW_L2_TO_L1 = os.path.join(OUT_DIR, "l2_to_l1.pt")    # 👈 保存更新后的层级映射
OUT_NEW_TOOLS_DICT = os.path.join(OUT_DIR, "task1_tools_with_id.json")

os.makedirs(OUT_DIR, exist_ok=True)

# 核心常量
START_TOOL_ID = 11112  
EPSILON = 1e-4         
NUM_NEW_BOXES = 50     

def main():
    print("="*50)
    print("🚀 启动 IH-PromptDSI 层级物理空间扩容引擎 (L1 约束版)")
    print("="*50)

    # ==========================================
    # 📦 第一步：加载老教授的旧记忆
    # ==========================================
    old_l1_centers = torch.load(OLD_L1_CENTERS_PATH, weights_only=False).float()
    old_l2_centers = torch.load(OLD_L2_CENTERS_PATH, weights_only=False).float()
    old_l2_widths = torch.load(OLD_L2_WIDTHS_PATH, weights_only=False).float()
    old_tool_to_l2 = torch.load(OLD_TOOL_TO_L2_PATH, weights_only=False)
    old_l2_to_l1 = torch.load(OLD_L2_TO_L1_PATH, weights_only=False)
    
    old_num_boxes = old_l2_centers.shape[0]
    num_l1 = old_l1_centers.shape[0]
    dim = old_l2_centers.shape[1]
    print(f">> 🔒 成功加载旧空间: {num_l1} 个固定 L1 大类, {old_num_boxes} 个 L2 盒子")

    # ==========================================
    # 🆔 第二步：解析 ToolGen 格式，派发全新全局 ID
    # ==========================================
    with open(TASK1_RAW_TOOLS_JSON, "r", encoding="utf-8") as f:
        new_raw_tools = json.load(f)
        
    task1_tools_with_id = {}
    tool_texts_for_encoder = []
    
    current_tool_id = START_TOOL_ID
    # 用 set 去重，防止同一个 API 有多条训练数据
    seen_anchors = set() 
    
    print(">> 🔍 正在从对话记录中精准解析工具特征...")
    for item in new_raw_tools:
        user_content, assistant_content = "", ""
        for conv in item.get("conversations", []):
            if conv.get("role") == "user":
                user_content = conv.get("content", "")
            elif conv.get("role") == "assistant":
                assistant_content = conv.get("content", "").strip()
                
        # 1. 提取真名锚点
        match_ast = re.match(r'<<(.+?)&&(.+?)>>', assistant_content)
        if not match_ast: continue
        anchor = f"<<{match_ast.group(1).strip()}&&{match_ast.group(2).strip()}>>"
        
        if anchor in seen_anchors: continue
        seen_anchors.add(anchor)
        
        # 2. 从 User 提取文本
        regex = r"Tool Name:\s*(.*?)\s*Tool Description:\s*(.*?)\s*Api Name:\s*(.*?)\s*Api Description:\s*(.*)"
        match_user = re.search(regex, user_content, re.DOTALL)
        
        if match_user:
            t_name = match_user.group(1).strip()
            t_desc = match_user.group(2).strip()
            a_name = match_user.group(3).strip()
            a_desc = match_user.group(4).strip()
            clean_text = f"Tool: {t_name}. Description: {t_desc}. API: {a_name}. API Description: {a_desc}"
        else:
            # 如果正则没匹配上，用基础名字兜底
            clean_text = f"Tool: {match_ast.group(1).strip()}. API: {match_ast.group(2).strip()}."
            
        task1_tools_with_id[str(current_tool_id)] = clean_text
        tool_texts_for_encoder.append(clean_text)
        current_tool_id += 1

    num_new_tools = len(tool_texts_for_encoder)
    print(f">> 🆔 提取完毕！共获得 {num_new_tools} 个独立新工具，ID: {START_TOOL_ID} ~ {current_tool_id - 1}")

    # ==========================================
    # 🧠 第三步：提取特征
    # ==========================================
    print(">> 🧠 正在加载 SentenceTransformer 提取 384 维特征...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")
    new_vectors = encoder.encode(tool_texts_for_encoder, convert_to_tensor=True, show_progress_bar=True).cpu()

    # ==========================================
    # 🏗️ 第四步：划定新盒子，并将其归类到固定的 50 个 L1 中
    # ==========================================
    actual_new_boxes = min(NUM_NEW_BOXES, num_new_tools)
    kmeans = KMeans(n_clusters=actual_new_boxes, random_state=42, n_init=10)
    new_labels = kmeans.fit_predict(new_vectors.numpy()) 
    
    unique_labels = np.unique(new_labels)
    valid_new_boxes = len(unique_labels)
    print(f">> 🏗️ 成功聚出 {valid_new_boxes} 个新 L2 盒子，正在进行 L1 归属映射...")

    new_centers = torch.zeros((valid_new_boxes, dim), dtype=torch.float32)
    new_widths = torch.zeros((valid_new_boxes, dim), dtype=torch.float32)
    
    new_tool_to_l2 = old_tool_to_l2.copy() 
    new_l2_to_l1 = old_l2_to_l1.copy() # 复制旧映射
    
    for i, cluster_idx in enumerate(unique_labels):
        indices = np.where(new_labels == cluster_idx)[0]
        cluster_vecs = new_vectors[indices]
        
        max_v = torch.max(cluster_vecs, dim=0)[0]
        min_v = torch.min(cluster_vecs, dim=0)[0]
        
        center_v = (max_v + min_v) / 2.0
        new_centers[i] = center_v
        new_widths[i] = (max_v - min_v) / 2.0 + EPSILON
        
        global_box_id = old_num_boxes + i
        
        # 🎯 核心 L1 归属判定：计算当前 L2 Center 到 50 个 L1 Centers 的欧氏距离
        # 将 L2 center 扩展维度以便广播计算距离
        dist_to_l1 = torch.norm(old_l1_centers - center_v.unsqueeze(0), dim=1)
        closest_l1_id = torch.argmin(dist_to_l1).item() # 找到最近的 L1 大类 ID
        
        # 将这个新诞生的 L2 盒子，正式挂载到最近的旧 L1 名下
        new_l2_to_l1[global_box_id] = closest_l1_id
        
        # 分配工具到该 L2 Box
        for idx in indices:
            global_tool_id = START_TOOL_ID + idx
            new_tool_to_l2[global_tool_id] = global_box_id

    # ==========================================
    # 🧬 第五步：终极物理缝合
    # ==========================================
    print(">> 🧬 正在将新世界与旧世界进行物理级缝合...")
    combined_centers = torch.cat([old_l2_centers, new_centers], dim=0)
    combined_widths = torch.cat([old_l2_widths, new_widths], dim=0)

    # 存盘输出
    torch.save(combined_centers, OUT_NEW_CENTERS)
    torch.save(combined_widths, OUT_NEW_WIDTHS)
    torch.save(new_tool_to_l2, OUT_NEW_TOOL_TO_L2)
    torch.save(new_l2_to_l1, OUT_NEW_L2_TO_L1)  # 👈 存入新的层级映射
    with open(OUT_NEW_TOOLS_DICT, "w", encoding="utf-8") as f:
        json.dump(task1_tools_with_id, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*50)
    print(f"🎉 成功！所有新工具已被完美限制在原有的 50 个 L1 空间内！")
    print(f"   -> 物理盒子总数从 {old_num_boxes} 扩容到了 {combined_centers.shape[0]}")
    print(f"   -> 新增了 {valid_new_boxes} 条 L2 -> L1 的层级约束映射")
    print("="*50)

if __name__ == "__main__":
    main()