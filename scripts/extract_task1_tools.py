import torch
import os
import json
from sentence_transformers import SentenceTransformer

# ==========================================================
# 🚨 路径配置
# ==========================================================
TASK1_TOOLS_JSON = "./data/task1/raw/task1_tools_with_id.json"
OUT_DIR = "./data/task1/features"
OUT_FEATURES_PATH = os.path.join(OUT_DIR, "task1_tool_features.pt")

os.makedirs(OUT_DIR, exist_ok=True)

def main():
    print("="*50)
    print("⚡ 启动 Task 1 [384维轻量特征榨汁机]")
    print("="*50)
    
    if not os.path.exists(TASK1_TOOLS_JSON):
        raise FileNotFoundError(f"找不到 {TASK1_TOOLS_JSON}！")
        
    with open(TASK1_TOOLS_JSON, 'r', encoding='utf-8') as f:
        task1_tools = json.load(f)
        
    tool_texts = list(task1_tools.values())
    
    # 加载 384 维度的轻量级提取器 (大概率就是这个模型)
    # 如果你本地没联网，可以把它换成你本地的路径，比如 "/data/wyx/models/all-MiniLM-L6-v2"
    print(">> 🧠 正在加载 SentenceTransformer (all-MiniLM-L6-v2)...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"❌ 加载模型失败，如果你在离线服务器，请把 'all-MiniLM-L6-v2' 改成本地模型路径！\n错误信息: {e}")
        return

    print(f">> 🚀 开始提取 {len(tool_texts)} 个新工具的 384 维特征...")
    
    # 直接一键提取，极其丝滑
    embeddings = model.encode(tool_texts, convert_to_tensor=True, show_progress_bar=True)
    
    # 存盘
    torch.save(embeddings.cpu(), OUT_FEATURES_PATH)
    
    print("\n" + "="*50)
    print(f"✅ 提取完成！特征形状: {embeddings.shape} (完美对应 [样本数, 384])")
    print(f"📂 已覆盖保存至: {OUT_FEATURES_PATH}")
    print("="*50)

if __name__ == "__main__":
    main()