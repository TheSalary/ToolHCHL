import json
import os
import re

input_path = "/home/wyx/ToolPrompt/data/toolgen_atomic_memorization.json"
tools_root = "/home/wyx/toolbench/data/toolenv/tools"
output_path = "/home/wyx/ToolPrompt/data/toolgen_atomic_memorization_extracted.json"

def normalize(name):
    # 更暴力的标准化：剔除所有非字母和非数字的字符，全部转小写
    # 例如 "Thai Driver's License OCR" -> "thaidriverslicenseocr"
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

# 1. 建立 L2 (Tool) 到 L1 (Category/Domain) 的映射表
l2_to_l1 = dict()
if os.path.exists(tools_root):
    for l1 in os.listdir(tools_root):
        l1_path = os.path.join(tools_root, l1)
        if not os.path.isdir(l1_path):
            continue
        for l2 in os.listdir(l1_path):
            l2_path = os.path.join(l1_path, l2)
            if os.path.isdir(l2_path):
                l2_to_l1[normalize(l2)] = l1
else:
    print(f"警告: 未找到文件夹 {tools_root}，请检查路径。")

# 2. 读取原始数据
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

results = []
for item in data:
    conversations = item.get("conversations", [])
    user_content = ""
    assistant_content = ""
    
    # 1. 拿到两段文本
    for conv in conversations:
        if conv.get("role") == "user":
            user_content = conv.get("content", "")
        elif conv.get("role") == "assistant":
            assistant_content = conv.get("content", "")
            
    # 2. 从 assistant 精准提取名字 (大类下的小类名)
    tool_name = None
    match_assistant = re.match(r"<<(.+?)&&", assistant_content)
    if match_assistant:
        tool_name = match_assistant.group(1).strip()
    
    # 3. 从 user 文本中精准切割四个字段
    # 使用正则表达式匹配这四个固定的锚点
    regex_pattern = r"Tool Name:\s*(.*?)\s*Tool Description:\s*(.*?)\s*Api Name:\s*(.*?)\s*Api Description:\s*(.*)"
    match_user = re.search(regex_pattern, user_content, re.DOTALL)
    
    api_desc_clean = ""
    if match_user:
        # 我们真正需要的是最干净的 API 描述
        api_desc_clean = match_user.group(4).strip()
    else:
        # 如果正则没匹配上，说明这批数据格式异常，直接跳过或者记录
        continue

    # 4. 匹配 Category (L1 Box)
    category = None
    if tool_name:
        norm_tool_name = normalize(tool_name)
        category = l2_to_l1.get(norm_tool_name)
        
    # 5. 【极其重要】处理 null 的情况
    if category is None:
        # 方案 A: 直接丢弃 (推荐！保证物理空间的绝对纯净)
        # continue 
        
        # 方案 B: 归入一个统一的 "Miscellaneous" (杂项) L1 Box
        category = "Miscellaneous" 
        
    results.append({
        "toolname": tool_name,
        "api_description": api_desc_clean, # 现在这里绝对干净了！
        "category": category
    })
# 3. 保存结果
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"成功提取 {len(results)} 个工具意图，结果已保存至 {output_path}")