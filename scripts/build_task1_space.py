import os
import json
import torch
from sklearn.cluster import KMeans
import numpy as np

# ================= 路径配置 =================
OLD_L2_CENTERS_PATH = "./data/train/clusters/l2_centers.pt"
TASK1_RAW_TOOL_TXT = "./data/task1/raw/tool.txt"

# 输出路径
OUT_TASK1_DIR = "./data/task1/clusters"
OUT_TASK1_TOOLS_JSON = "./data/task1/raw/task1_tools_with_id.json"
OUT_NEW_L2_CENTERS = os.path.join(OUT_TASK1_DIR, "l2_centers.pt")
OUT_NEW_TOOL_TO_L2 = os.path.join(OUT_TASK1_DIR, "tool_to_l2.pt")

os.makedirs(OUT_TASK1_DIR, exist_ok=True)

def main():
    print("="*50)
    print("🚀 开始 Task 1 物理盒子外科扩建手术...")
    print("="*50)

    # 1. 加载老教授的旧记忆 (582个老盒子)
    old_l2_centers = torch.load(OLD_L2_CENTERS_PATH, weights_only=False).float()
    old_num_boxes = old_l2_centers.shape[0]
    dim = old_l2_centers.shape[1]
    print(f">> 📦 成功加载旧物理空间: 包含 {old_num_boxes} 个盒子，维度 {dim}")

    # 2. 读取 Task 1 新工具并分配全局 ID
    with open(TASK1_RAW_TOOL_TXT, 'r', encoding='utf-8') as f:
        new_tools_text = [line.strip() for line in f if line.strip()]
    
    num_new_tools = len(new_tools_text)
    start_id = 11112 # 基座是 0 ~ 11111
    
    task1_tools_with_id = {}
    for i, txt in enumerate(new_tools_text):
        global_id = start_id + i
        task1_tools_with_id[str(global_id)] = txt
        
    with open(OUT_TASK1_TOOLS_JSON, 'w', encoding='utf-8') as f:
        json.dump(task1_tools_with_id, f, indent=2)
    print(f">> 🆔 ID 分配完毕！共 {num_new_tools} 个新工具，ID 范围: {start_id} ~ {start_id + num_new_tools - 1}")

    # 3. 提取新工具的特征 (⚠️ 这里需要你接入原来的特征提取逻辑)
    print(">> 🧠 正在提取新工具的 4096 维特征...")
    new_features = torch.load("./data/task1/features/task1_tool_features.pt", weights_only=False).float()
    
    # 4. 对新工具进行增量聚类 (假设我们给新工具建 50 个新盒子)
    num_new_boxes = min(50, num_new_tools) # 如果新工具很少，就少建几个盒子
    print(f">> 🏗️ 正在为新工具划定 {num_new_boxes} 个专属新盒子...")
    
    kmeans = KMeans(n_clusters=num_new_boxes, random_state=42, n_init=10)
    new_labels = kmeans.fit_predict(new_features.numpy())
    new_centers = torch.tensor(kmeans.cluster_centers_).float()
    
    # 映射字典: tool_id -> l2_box_id
    # 注意！新盒子的 ID 必须顺延旧盒子 (比如从 582 开始)
    new_tool_to_l2 = {}
    for i, label in enumerate(new_labels):
        global_id = start_id + i
        global_box_id = old_num_boxes + label # 582 + label
        new_tool_to_l2[global_id] = global_box_id

    # 5. 终极缝合：将新盒子拼接在老盒子后面！
    combined_l2_centers = torch.cat([old_l2_centers, new_centers], dim=0)
    
    # 保存最新战果
    torch.save(combined_l2_centers, OUT_NEW_L2_CENTERS)
    torch.save(new_tool_to_l2, OUT_NEW_TOOL_TO_L2)
    
    print("="*50)
    print(f"🎉 手术成功！")
    print(f"   -> 物理盒子总数从 {old_num_boxes} 扩容到了 {combined_l2_centers.shape[0]}！")
    print(f"   -> 新空间已保存至: {OUT_NEW_L2_CENTERS}")
    print("="*50)

if __name__ == "__main__":
    main()