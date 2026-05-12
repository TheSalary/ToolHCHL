import json
import re
import os

# ==========================================
# 🚨 路径配置
# ==========================================
RETRIEVAL_RAW_PATH = "./data/task1/raw/retrieval_train.json"
TOOLS_WITH_ID_PATH = "./data/task1/raw/task1_tools_with_id.json"
OUT_RETRIEVAL_PATH = "./data/task1/raw/task1_retrieval_clean.json"

def normalize(name):
    """极其暴力的标准化：剔除所有非字母和非数字的字符，全部转小写"""
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

def main():
    print("="*50)
    print("🚀 启动 Task 1 题库 ID [暴力换血] 手术...")
    print("="*50)

    # 1. 加载我们刚刚分配好 ID 的工具字典
    with open(TOOLS_WITH_ID_PATH, 'r', encoding='utf-8') as f:
        tools_dict = json.load(f)

    # 建立反向映射: Tool Name -> Global ID (11112+)
    name_to_id = {}
    for tid, text in tools_dict.items():
        # 提取我们在上一个脚本里拼装的 Tool Name
        match = re.search(r"Tool Name:\s*(.*?)\.", text)
        if match:
            raw_name = match.group(1).strip()
            norm_name = normalize(raw_name) # 🔫 启动暴力清洗
            name_to_id[norm_name] = int(tid)

    # 2. 读取原始的检索题库
    with open(RETRIEVAL_RAW_PATH, 'r', encoding='utf-8') as f:
        retrieval_data = json.load(f)

    clean_retrieval_data = []
    hit_count = 0
    miss_count = 0

    # 3. 遍历题库，替换 ID
    for item in retrieval_data:
        query = ""
        target_tool = ""
        
        # 解析 ToolGen 格式的对话
        for conv in item.get("conversations", []):
            if conv.get("role") == "user":
                query = conv.get("content", "").strip()
            elif conv.get("role") == "assistant":
                # 提取目标工具名 <<ToolName&&
                match_ast = re.match(r"<<(.+?)&&", conv.get("content", ""))
                if match_ast:
                    raw_target = match_ast.group(1).strip()
                    target_tool = normalize(raw_target) # 🔫 启动暴力清洗
        
        # 如果找到了目标工具，并且在我们的新 ID 字典里
        if query and target_tool:
            if target_tool in name_to_id:
                global_id = name_to_id[target_tool]
                clean_retrieval_data.append({
                    "query": query,
                    "target_id": global_id,
                    "target_name": raw_target # 保存个原始名字留作纪念
                })
                hit_count += 1
            else:
                miss_count += 1

    # 4. 存盘！
    with open(OUT_RETRIEVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(clean_retrieval_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 换血完毕！成功匹配 {hit_count} 条训练数据！(未匹配跳过: {miss_count} 条)")
    print(f"📂 干净的终极题库已保存至: {OUT_RETRIEVAL_PATH}")
    print("="*50)

if __name__ == "__main__":
    main()