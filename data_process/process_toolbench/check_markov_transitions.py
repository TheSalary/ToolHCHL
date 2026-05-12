import torch
import json
import random

# --- 路径配置 ---
m_global_path = "/home/wyx/ToolPrompt/data/processed_real/m_global.pt"
tools_json_path = "/home/wyx/ToolPrompt/data/toolgen_format_balanced/train/train_tools_with_id.json"
tool_to_l2_path = "/home/wyx/ToolPrompt/data/processed_real/tool_to_l2.pt"

def check_transitions():
    print("正在加载空间矩阵与工具字典...\n")
    # 加载数据
    m_global = torch.load(m_global_path, weights_only=True)
    with open(tools_json_path, "r", encoding="utf-8") as f:
        tools = json.load(f)
    tool_to_l2 = torch.load(tool_to_l2_path, weights_only=True)

    num_tools, num_boxes = m_global.shape
    
    # 1. 构建字典：通过 tool_id 快速查找工具信息
    id_to_tool = {t["tool_id"]: t for t in tools}
    
    # 2. 构建字典：看看每个 L2 Box 里面都装了什么（取前几个作为代表）
    box_contents = {}
    for t in tools:
        tid = t["tool_id"]
        if tid in tool_to_l2:
            bid = tool_to_l2[tid]
            if bid not in box_contents:
                box_contents[bid] = []
            # 存入简短的描述方便查看
            box_contents[bid].append(f"[{t['l1_domain']}] {t['tool_name']}/{t['api_name']}")

    # 3. 过滤出真正有转移记录的工具
    # 如果一行全都是均匀分布 (1 / num_boxes)，说明这个工具在轨迹里没作为起点出现过
    uniform_prob = 1.0 / num_boxes
    valid_tool_ids = []
    for i in range(num_tools):
        # 如果最大概率明显大于平均分配的概率，说明它有确定的转移偏好
        if m_global[i].max().item() > uniform_prob + 1e-4:
            valid_tool_ids.append(i)

    print(f"在 {num_tools} 个工具中，有 {len(valid_tool_ids)} 个工具在轨迹中拥有明确的下一步跳转方向。\n")
    print("="*60)

    # 4. 随机抽样 5 个工具展示 (你也可以自己指定 ID)
    sample_ids = random.sample(valid_tool_ids, min(5, len(valid_tool_ids)))
    
    for tid in sample_ids:
        source_tool = id_to_tool[tid]
        print(f"🟢 【上一步动作】 源工具 [ID: {tid}]:")
        print(f"   => {source_tool.get('raw_intent', source_tool.get('api_description'))}")
        
        # 获取这一行的概率分布
        probs = m_global[tid]
        
        # 找出概率最高的前 3 个目的地 Box
        topk_probs, topk_indices = torch.topk(probs, 3)
        
        print(f"   👇 【大概率会去哪？】 预测的下一步 Top-3 盒子:")
        for i in range(3):
            box_id = topk_indices[i].item()
            prob = topk_probs[i].item()
            
            # 如果概率极低，就不展示了
            if prob < 0.01:
                continue
                
            print(f"      [{i+1}] 目的地 Box ID: {box_id} (转移概率: {prob:.1%})")
            
            # 打印这个盒子里面的代表性工具，让你看看合不合理
            sample_tools_in_box = box_contents.get(box_id, [])[:3] # 最多展示3个
            print(f"          该盒子包含: {sample_tools_in_box} ...等")
            
        print("-" * 60)

if __name__ == "__main__":
    check_transitions()