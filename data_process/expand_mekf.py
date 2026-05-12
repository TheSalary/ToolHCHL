import torch
import json
import os

OLD_M_GLOBAL = "./data/train/clusters/m_global.pt"
NEW_TOOLS_JSON = "./data/task1/clusters/task1_tools_with_id.json"
OUT_M_GLOBAL = "./data/task1/clusters/m_global.pt"

def patch_m_global():
    print(">> 正在加载旧的马尔可夫转移矩阵...")
    old_m = torch.load(OLD_M_GLOBAL, weights_only=False)
    old_num_tools, old_num_boxes = old_m.shape
    
    with open(NEW_TOOLS_JSON, 'r', encoding='utf-8') as f:
        new_tools_dict = json.load(f)
    num_new_tools = len(new_tools_dict)
    
    total_tools = old_num_tools + num_new_tools
    total_boxes = old_num_boxes + 50  # 582 + 50 = 632
    
    print(f">> 扩容目标: 工具数 {old_num_tools} -> {total_tools}, 盒子数 {old_num_boxes} -> {total_boxes}")
    
    # 建立一个全 0 的新大矩阵
    new_m = torch.zeros((total_tools, total_boxes), dtype=old_m.dtype)
    
    # 把老矩阵原封不动地贴在左上角
    new_m[:old_num_tools, :old_num_boxes] = old_m
    
    # 对新增加的工具，由于没有历史轨迹，赋予它们去往任何盒子的均匀先验概率
    new_m[old_num_tools:, :] = 1.0 / total_boxes
    
    torch.save(new_m, OUT_M_GLOBAL)
    print(f"✅ 补丁搞定！新的 m_global.pt 已保存至 {OUT_M_GLOBAL}")

if __name__ == "__main__":
    patch_m_global()