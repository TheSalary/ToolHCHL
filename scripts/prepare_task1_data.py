import json
import re
import os

# ==========================================
# 🚨 路径配置
# ==========================================
INPUT_PATH = "./data/task1/raw/memorization_train.json"
OUT_JSON_PATH = "./data/task1/raw/task1_tools_with_id.json"
OUT_TXT_PATH = "./data/task1/raw/task1_clean_tools.txt" # 顺手存个纯文本版备查

START_ID = 11112 # 基座工具是从 0 到 11111

def clean_and_format_tool(content):
    """用你当时的硬核正则，精准切割工具属性"""
    regex_pattern = r"Tool Name:\s*(.*?)\s*Tool Description:\s*(.*?)\s*Api Name:\s*(.*?)\s*Api Description:\s*(.*)"
    match_user = re.search(regex_pattern, content, re.DOTALL)
    
    if match_user:
        tool_name = match_user.group(1).strip()
        tool_desc = match_user.group(2).strip()
        api_name = match_user.group(3).strip()
        api_desc = match_user.group(4).strip()
        
        # 拼装成给大模型提取特征的干净文本 (格式你可以按需微调)
        clean_text = f"Tool Name: {tool_name}. Description: {tool_desc}. API: {api_name}. API Description: {api_desc}"
        return clean_text
    return None

def main():
    print("="*50)
    print("🚀 启动 Task 1 数据清洗与 ID 继任仪式")
    print("="*50)

    if not os.path.exists(INPUT_PATH):
        print(f"❌ 找不到输入文件: {INPUT_PATH}")
        return

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    task1_tools_with_id = {}
    clean_texts_list = []
    current_id = START_ID
    
    # 用集合去重，防止 task1 里有重复的工具
    seen_texts = set()

    for item in data:
        conversations = item.get("conversations", [])
        for conv in conversations:
            if conv.get("role") == "user":
                content = conv.get("content", "")
                clean_text = clean_and_format_tool(content)
                
                if clean_text and clean_text not in seen_texts:
                    seen_texts.add(clean_text)
                    task1_tools_with_id[str(current_id)] = clean_text
                    clean_texts_list.append(clean_text)
                    current_id += 1

    # 保存带 ID 的 JSON (给特征提取脚本用)
    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(task1_tools_with_id, f, ensure_ascii=False, indent=2)

    # 保存纯文本版 (给你自己肉眼检查用的)
    with open(OUT_TXT_PATH, "w", encoding="utf-8") as f:
        for txt in clean_texts_list:
            f.write(txt + "\n")

    print(f"✅ 清洗完毕！成功提取 {len(task1_tools_with_id)} 个独立的新工具！")
    print(f"🆔 分配的 ID 范围: {START_ID} ~ {current_id - 1}")
    print(f"📂 核心数据已保存至: {OUT_JSON_PATH}")
    print("="*50)

if __name__ == "__main__":
    main()