import json
import torch
import numpy as np
import os
import re
from glob import glob
from tqdm import tqdm

# --- 路径配置 ---
# 85% 训练集工具信息
tools_json_path = "/home/wyx/ToolPrompt/data/toolgen_format_balanced/train/train_tools_with_id.json"
# 物理空间映射表
tool_to_l2_path = "/home/wyx/ToolPrompt/data/processed_real/tool_to_l2.pt"
# ToolBench 轨迹文件的根目录 (你需要填入真实的轨迹文件夹路径)
# 例如: "/home/wyx/toolbench/data/instruction/G1_instruction/"
trajectories_dirs = [
    "/home/wyx/toolbench/data/answer/G1_answer",
    "/home/wyx/toolbench/data/answer/G2_answer",
    "/home/wyx/toolbench/data/answer/G3_answer"
] # 请确认轨迹 JSON 存在哪里
output_matrix_path = "/home/wyx/ToolPrompt/data/processed_real/m_global.pt"

def normalize_name(name):
    """暴力去除所有非字母数字，转小写，用于绝对匹配"""
    return re.sub(r'[^a-zA-Z0-9]', '', str(name)).lower()

def build_markov_matrix():
    print("1. 正在加载工具字典与空间映射...")
    with open(tools_json_path, "r", encoding="utf-8") as f:
        tools = json.load(f)
    
    tool_to_l2 = torch.load(tool_to_l2_path)
    
    # 构建快速查找字典：通过 "apifortool" 查找 tool_id
    name_to_id = {}
    for t in tools:
        t_id = t["tool_id"]
        # Toolbench 轨迹里的格式通常是 api_for_tool
        api_norm = normalize_name(t["api_name"])
        tool_norm = normalize_name(t["tool_name"])
        # 组装成极简字符串，例如 "checkhealthforsquake"
        combined_key = f"{api_norm}for{tool_norm}"
        name_to_id[combined_key] = t_id

    num_tools = len(tools)
    # 获取最大的 l2 box ID，确定矩阵列数
    num_l2_boxes = max(tool_to_l2.values()) + 1
    
    print(f"   -> 训练集工具数: {num_tools}, L2 Box 数: {num_l2_boxes}")
    
    # 初始化转移计数矩阵
    transition_counts = np.zeros((num_tools, num_l2_boxes), dtype=np.float32)

    # 2. 扫描轨迹文件
    print("2. 正在解析轨迹文件，提取马尔可夫链...")
    # 假设轨迹是以 .json 结尾的文件，这里递归查找目录下所有 json
    # 注意：如果你的轨迹不在这个目录，请修改 glob 的路径
    traj_files = []
    for d in trajectories_dirs:
        if os.path.exists(d):
            traj_files.extend(glob(os.path.join(d, "**/*.json"), recursive=True))
        else:
            print(f"警告：找不到目录 {d}")
    
    valid_transitions = 0
    skipped_unk = 0

    for file_path in tqdm(traj_files, desc="解析轨迹"):
        # 排除我们自己生成的元数据文件
        if "train_tools" in file_path or "processed" in file_path:
            continue
            
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            continue
            
        # 确保是包含 answer_generation 的合法轨迹
        if "answer_generation" not in data or "train_messages" not in data["answer_generation"]:
            continue
            
        messages_list = data["answer_generation"]["train_messages"]
        if not messages_list:
            continue
            
        # 获取最后一次完整的尝试
        final_attempt = messages_list[-1]
        
        # 提取调用链
        chain = []
        for msg in final_attempt:
            if msg.get("role") == "assistant" and "function_call" in msg:
                func_name = msg["function_call"].get("name", "")
                if func_name and func_name.lower() != "finish":
                    # 将 checkhealth_for_squake 转成 checkhealthforsquake
                    chain.append(normalize_name(func_name))
        
        # 如果链条长度小于 2，就没有转移边
        if len(chain) < 2:
            continue
            
        # 3. 统计转移边
        for i in range(len(chain) - 1):
            func_prev = chain[i]
            func_next = chain[i+1]
            
            # 解决未知工具 (UNK) 问题：
            if func_prev not in name_to_id or func_next not in name_to_id:
                skipped_unk += 1
                continue # 丢弃包含未知工具的边
                
            prev_id = name_to_id[func_prev]
            next_id = name_to_id[func_next]
            
            # 获取 next_id 对应的 L2 Box
            if next_id in tool_to_l2:
                next_l2_box = tool_to_l2[next_id]
                transition_counts[prev_id, next_l2_box] += 1
                valid_transitions += 1

    # 4. 矩阵归一化 (计算先验概率)
    print("3. 正在将频次归一化为转移概率矩阵...")
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    
    # 避免除以 0：如果某个工具从未作为上一步出现过，我们给它一个均匀分布，或者保持全 0
    # 这里我们采用平均分配，表示“如果没见过它转移，那么去哪个盒子都有可能”
    zero_rows = (row_sums == 0).flatten()
    row_sums[zero_rows] = 1.0 # 临时设为1防止报错
    
    m_global = transition_counts / row_sums
    
    # 把没有记录的行变成均匀分布
    m_global[zero_rows] = 1.0 / num_l2_boxes
    
    m_global_tensor = torch.tensor(m_global, dtype=torch.float32)

    # 5. 保存结果
    torch.save(m_global_tensor, output_matrix_path)
    print(f"\n=== M_global 矩阵构建完成 ===")
    print(f"矩阵形状: {m_global_tensor.shape} (N_tools x N_boxes)")
    print(f"成功记录的合法转移边: {valid_transitions}")
    print(f"因涉及 15% 未知工具而丢弃的边: {skipped_unk}")
    print(f"马尔可夫先验图已保存至: {output_matrix_path}")

if __name__ == "__main__":
    build_markov_matrix()