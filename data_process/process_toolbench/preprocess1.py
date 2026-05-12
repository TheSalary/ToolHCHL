import json
import re

input_path = "ToolPrompt/data/toolgen_format_balanced/train/memorization_train.json"
output_path = "ToolPrompt/data/toolgen_format_balanced/train/memorization_extracted.json"

def extract_tool_info(content):
    # 提取 Tool Name
    toolname = None
    tooldescription = None
    # Tool Name: xxx. Tool Description: xxx.
    match_name = re.search(r'Tool Name:\s*([^\.]+)', content)
    if match_name:
        toolname = match_name.group(1).strip()
    match_desc = re.search(r'Tool Description:\s*([^\.]+)', content)
    if match_desc:
        tooldescription = match_desc.group(1).strip()
    return toolname, tooldescription

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

results = []
for item in data:
    conversations = item.get("conversations", [])
    for conv in conversations:
        if conv.get("role") == "user":
            content = conv.get("content", "")
            toolname, tooldescription = extract_tool_info(content)
            results.append({
                "content": content,
                "toolname": toolname,
                "tooldescription": tooldescription
            })

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"提取完成，已保存到 {output_path}")