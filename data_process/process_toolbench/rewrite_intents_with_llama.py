import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm  # 用于显示进度条

# --- 路径配置 ---
input_json_path = "/home/wyx/ToolPrompt/data/toolgen_format_balanced/train/train_tools_with_id.json"
output_json_path = "/home/wyx/ToolPrompt/data/toolgen_format_balanced/train/train_tools_rewritten.json"
local_llama_path = "/data/wyx/llama3-8b/"  # 你的本地 Llama3 路径

def load_llama3():
    print(f"正在以 bfloat16 精度加载 Llama-3 (强制使用单张 4090 完整加载，避免跨卡 Bug)...")
    tokenizer = AutoTokenizer.from_pretrained(local_llama_path, trust_remote_code=True)
    
    # 修复 pad_token
    if tokenizer.pad_token is None:
        if "<|end_of_text|>" in tokenizer.vocab:
            tokenizer.pad_token = "<|end_of_text|>"
        else:
            tokenizer.pad_token = tokenizer.eos_token

    # 【核心修改】：放弃 auto，直接强制把整个模型放进 cuda:0
    model = AutoModelForCausalLM.from_pretrained(
        local_llama_path,
        torch_dtype=torch.bfloat16,
        device_map={'': 'cuda:0'},  # 强制绑定到单一设备
        trust_remote_code=True
    )
    return tokenizer, model

def generate_intent(tokenizer, model, tool_name, api_name, description):
    prompt = f"""Rewrite the API description into a short, concise intent starting with a verb (Action + Business Object). Do not output any conversational text.

Example 1:
Tool Name: Estrelabet Aviator API
API Name: get_results
Description: This endpoint allows you to retrieve the latest results of the Aviator game on the Estrelabet.
Output: Retrieve latest Aviator game results

Example 2:
Tool Name: LeagueOfLegends
API Name: getChampions
Description: Retrieve all champions
Output: Retrieve all champions

Example 3 (Missing Description):
Tool Name: Thai Drivers License OCR
API Name: extract_info
Description: 
Output: Extract Thai driver license information

Target:
Tool Name: {tool_name}
API Name: {api_name}
Description: {description}
Output:"""

    # 【核心修改】：明确指定发送到 cuda:0
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.1, 
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        rewritten_action = full_response.split("Output:")[-1].strip()
        rewritten_action = rewritten_action.split('\n')[0].strip(' .,"\'')
    except:
        rewritten_action = description[:50] 
        
    return rewritten_action

def main():
    if not os.path.exists(input_json_path):
        print(f"找不到输入文件: {input_json_path}")
        return

    with open(input_json_path, "r", encoding="utf-8") as f:
        tools = json.load(f)

    tokenizer, model = load_llama3()

    print(f"\n开始使用 Llama-3 重写 {len(tools)} 个工具的意图...")
    
    # 使用 tqdm 包装，让你在终端能看到处理进度条和预计剩余时间
    for tool in tqdm(tools, desc="意图重写进度"):
        t_name = tool.get("tool_name", "")
        a_name = tool.get("api_name", "")
        desc = tool.get("api_description", "")
        l1_domain = tool.get("l1_domain", "Miscellaneous")
        
        # 即使没有 description，大模型也会根据名字生成
        action_object = generate_intent(tokenizer, model, t_name, a_name, desc)
        
        # 严格拼接为 [领域标签] + 动作 + 核心业务对象
        final_intent = f"[{l1_domain}] {action_object}"
        
        tool["rewritten_intent"] = final_intent

    print("\n重写完成！正在保存文件...")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(tools, f, ensure_ascii=False, indent=2)
        
    print(f"标准化的意图数据已保存至: {output_json_path}")

if __name__ == "__main__":
    # 如果你没有安装 tqdm，运行前在终端执行：pip install tqdm
    main()