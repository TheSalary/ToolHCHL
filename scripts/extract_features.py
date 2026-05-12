import torch
import os
import warnings
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
warnings.filterwarnings("ignore")

# 引入你的数据集
from data_process.dataset import get_dataloader

LLAMA_PATH = "/data/wyx/llama3-8b"
# 🚀 纯推理没有梯度，极其省显存！Batch Size 直接拉到 32 起飞
BATCH_SIZE = 32 

def extract_and_save(base_llm, tokenizer, dataloader, device, save_path="offline_features.pt"):
    base_llm.eval()
    all_q_vecs = []
    all_l2_targets = []
    all_l1_targets = []
    all_tool_labels = [] # 把终极工具标签也顺手存下来，以防万一

    print("\n🚀 开始离线提取大模型 Layer 1 特征...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Features"):
            query_texts = batch["query_text"]
            
            # 为了省内存，标签直接放 CPU
            target_l2 = batch["target_l2_id"].cpu()
            target_l1 = batch["target_l1_id"].cpu()
            tool_label = batch["tool_label"].cpu()

            # 1. 文本转 Token
            inputs = tokenizer(query_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            
            # 2. 跑过 Llama-3 的第一层
            outputs = base_llm(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            layer_1_hidden = outputs.hidden_states[1]

            # 3. AVG 池化 (扣除 Padding)
            mask_expanded = inputs.attention_mask.unsqueeze(-1).expand(layer_1_hidden.size()).float()
            sum_hidden = torch.sum(layer_1_hidden * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

            # 4. 拿到 4096 维的最原始特征，转回 bfloat16
            q_vec_avg = sum_hidden / sum_mask
            q_vec_avg = q_vec_avg.to(torch.bfloat16)

            # 5. 存入内存列表 (放到 CPU，防止显存爆炸)
            all_q_vecs.append(q_vec_avg.cpu())
            all_l2_targets.append(target_l2)
            all_l1_targets.append(target_l1)
            all_tool_labels.append(tool_label)

    # 将所有的 Batch 拼成一个巨大的字典
    dataset_dict = {
        "q_vecs": torch.cat(all_q_vecs, dim=0),
        "l2_targets": torch.cat(all_l2_targets, dim=0),
        "l1_targets": torch.cat(all_l1_targets, dim=0),
        "tool_labels": torch.cat(all_tool_labels, dim=0)
    }

    # 一把保存到硬盘
    torch.save(dataset_dict, save_path)
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    
    print("\n" + "="*50)
    print(f"✅ 提取完成！共保存 {dataset_dict['q_vecs'].shape[0]} 条数据的浓缩特征。")
    print(f"📂 硬盘文件大小: {file_size_mb:.2f} MB")
    print(f"📊 核心特征维度: {dataset_dict['q_vecs'].shape} (完美对应 [样本数, 4096])")
    print("="*50)

if __name__ == "__main__":
    print("="*50)
    print("⚡ 启动 IH-PromptDSI [第一阶段：特征榨汁机]")
    print("="*50)
    
    # 只需要单卡即可完成
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(">> 正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(">> 正在以 bfloat16 精度加载 Llama-3 主干网络...")
    base_llm = AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH, torch_dtype=torch.bfloat16
    ).to(device)
    base_llm.eval()
    
    print(f">> 正在加载 Dataloader (提速模式 Batch Size = {BATCH_SIZE})...")
    dataloader = get_dataloader(batch_size=BATCH_SIZE)
    
    # 启动提取
    extract_and_save(base_llm, tokenizer, dataloader, device, save_path="offline_features.pt")