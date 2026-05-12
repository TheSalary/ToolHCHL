import json
import re
import os

# ==========================================
# 🚨 路径配置 (全部在 task1 目录下)
# ==========================================
MEMO_RAW = "./data/task1/raw/memorization_train.json"
RETRIEVAL_RAW = "./data/task1/raw/retrieval_train.json"

OUT_TOOLS_JSON = "./data/task1/raw/task1_tools_with_id.json"
OUT_RETRIEVAL_CLEAN = "./data/task1/raw/task1_retrieval_clean.json"

START_ID = 11112

def main():
    print("="*50)
    print("🚀 启动 Task 1 终极数据流水线 (基于 Assistant 真名锚点)")
    print("="*50)

    # ---------------------------------------------------------
    # 第一步：从 memorization 中提取真名，派发 ID
    # ---------------------------------------------------------
    with open(MEMO_RAW, "r", encoding="utf-8") as f:
        memo_data = json.load(f)

    task1_tools_with_id = {}
    target_to_id = {} # 核心映射表: "<<Tool&&Api>>" -> ID
    current_id = START_ID

    for item in memo_data:
        user_content, assistant_content = "", ""
        for conv in item.get("conversations", []):
            if conv.get("role") == "user":
                user_content = conv.get("content", "")
            elif conv.get("role") == "assistant":
                assistant_content = conv.get("content", "").strip()

        # 🎯 提取真名锚点
        match_ast = re.match(r'<<(.+?)&&(.+?)>>', assistant_content)
        if not match_ast:
            continue
            
        tool_name = match_ast.group(1).strip()
        api_name = match_ast.group(2).strip()
        true_target = f"<<{tool_name}&&{api_name}>>" # 绝对真名

        # 提取干净的 API 描述 (给你后面抽 384 维特征用)
        match_user = re.search(r"Tool Name:\s*(.*?)\s*Tool Description:\s*(.*?)\s*Api Name:\s*(.*?)\s*Api Description:\s*(.*)", user_content, re.DOTALL)
        api_desc = match_user.group(4).strip() if match_user else ""
        
        # 拼装干净文本用于特征榨汁
        clean_text = f"Tool: {tool_name}. API: {api_name}. Description: {api_desc}"

        # 派发 ID (去重)
        if true_target not in target_to_id:
            target_to_id[true_target] = current_id
            task1_tools_with_id[str(current_id)] = clean_text
            current_id += 1

    with open(OUT_TOOLS_JSON, "w", encoding="utf-8") as f:
        json.dump(task1_tools_with_id, f, ensure_ascii=False, indent=2)
    print(f"✅ 第一步完成：成功提取 {len(target_to_id)} 个独立工具！ID 范围: {START_ID} ~ {current_id-1}")

    # ---------------------------------------------------------
    # 第二步：清洗 Retrieval 题库，进行绝对对齐
    # ---------------------------------------------------------
    with open(RETRIEVAL_RAW, "r", encoding="utf-8") as f:
        ret_data = json.load(f)

    clean_retrieval_data = []
    hit_count, miss_count = 0, 0

    for item in ret_data:
        query, true_target = "", ""
        for conv in item.get("conversations", []):
            if conv.get("role") == "user":
                query = conv.get("content", "").strip()
            elif conv.get("role") == "assistant":
                # 同样的配方，同样的真名锚点
                match_ast = re.match(r'<<(.+?)&&(.+?)>>', conv.get("content", "").strip())
                if match_ast:
                    true_target = f"<<{match_ast.group(1).strip()}&&{match_ast.group(2).strip()}>>"
        
        if query and true_target:
            if true_target in target_to_id:
                clean_retrieval_data.append({
                    "query": query,
                    "target_id": target_to_id[true_target],
                    "target_name": true_target
                })
                hit_count += 1
            else:
                miss_count += 1

    with open(OUT_RETRIEVAL_CLEAN, "w", encoding="utf-8") as f:
        json.dump(clean_retrieval_data, f, ensure_ascii=False, indent=2)
        
    print(f"✅ 第二步完成：题库换血完毕！成功匹配 {hit_count} 条，未匹配 {miss_count} 条。")
    print("="*50)

if __name__ == "__main__":
    main()