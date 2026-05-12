import json
import os
import re

# --- 1. 路径配置 ---
original_json_path = "/home/wyx/ToolPrompt/data/toolgen_atomic_memorization.json"
tools_root = "/home/wyx/toolbench/data/toolenv/tools"
train_tools_txt_path = "/home/wyx/ToolPrompt/data/toolgen_format_balanced/train/tools.txt"
output_train_json_path = "/home/wyx/ToolPrompt/data/toolgen_format_balanced/train/train_tools_with_id.json"

def normalize(name):
    # 暴力标准化：去除非字母数字，用于文件夹匹配
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

def build_train_dataset():
    print("1. 正在读取 train/tools.txt 中的目标工具列表...")
    if not os.path.exists(train_tools_txt_path):
        print(f"找不到文件: {train_tools_txt_path}")
        return

    # 将 tools.txt 里的每一行（如 << xxx && yyy >>）存入集合，为了容错，去掉首尾空格
    with open(train_tools_txt_path, "r", encoding="utf-8") as f:
        valid_targets = set(line.strip() for line in f if line.strip())
    print(f"   -> 成功加载 {len(valid_targets)} 个目标调用标签。")

    print("2. 正在建立 L2 到 L1 (物理隔离墙) 的映射表...")
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
        print("   -> 警告: 未找到 tools 根目录，category 将全部标记为 Unknown。")

    print("3. 正在遍历原始数据，精准洗出 85% 的训练集并重新分配 ID...")
    with open(original_json_path, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    train_subset = []
    current_id = 0

    for item in original_data:
        conversations = item.get("conversations", [])
        user_content = ""
        assistant_content = ""
        
        for conv in conversations:
            if conv.get("role") == "user":
                user_content = conv.get("content", "")
            elif conv.get("role") == "assistant":
                assistant_content = conv.get("content", "").strip()
                
        # 核心判定：如果 assistant 的输出（如 <<Tool&&Api>>）不在我们的训练名单里，直接跳过！
        if assistant_content not in valid_targets:
            continue
            
        # 开始解析这个“被选中”的工具
        match_names = re.match(r'<<(.+?)&&(.+?)>>', assistant_content)
        if not match_names:
            continue
            
        tool_name = match_names.group(1).strip()
        api_name = match_names.group(2).strip()
        
        # 提取干净的 API 描述
        regex_pattern = r"Tool Name:\s*(.*?)\s*Tool Description:\s*(.*?)\s*Api Name:\s*(.*?)\s*Api Description:\s*(.*)"
        match_user = re.search(regex_pattern, user_content, re.DOTALL)
        
        api_desc_clean = ""
        if match_user:
            api_desc_clean = match_user.group(4).strip()
        
        # 匹配大类 (L1 Box)
        norm_tool_name = normalize(tool_name)
        category = l2_to_l1.get(norm_tool_name, "Miscellaneous") # 找不到的塞进杂项
        
        # 组装为后续 Encoder 和 Router 需要的终极格式
        new_tool = {
            "tool_id": current_id,
            "l1_domain": category,
            "tool_name": tool_name,
            "api_name": api_name,
            "api_description": api_desc_clean,
            # 将 L1 标签和清理后的描述拼接，这就是我们在几何空间里的“坐标指纹”
            "raw_intent": f"[{category}] {api_desc_clean}"
        }
        
        train_subset.append(new_tool)
        current_id += 1

    print("4. 正在保存带有全新连续 ID 的训练集子集...")
    os.makedirs(os.path.dirname(output_train_json_path), exist_ok=True)
    with open(output_train_json_path, "w", encoding="utf-8") as f:
        json.dump(train_subset, f, ensure_ascii=False, indent=2)

    print("\n=== 处理完成 ===")
    print(f"成功筛选并写入训练集: {len(train_subset)} 条数据 (已分配连续 ID: 0 ~ {len(train_subset)-1})")
    print(f"文件已就绪: {output_train_json_path}")

if __name__ == "__main__":
    build_train_dataset()