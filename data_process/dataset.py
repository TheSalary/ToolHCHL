import torch
import json
import os
import re
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm



def normalize_name(name):
    """暴力标准化，用于绝对匹配"""
    return re.sub(r'[^a-zA-Z0-9]', '', str(name)).lower()

class IH_ToolDataset(Dataset):
    def __init__(self, 
                 tools_json_path="./data/train/raw/train_tools_with_id.json",
                 tool_to_l2_path="./data/train/clusters/tool_to_l2.pt",
                 l2_to_l1_path="./data/train/clusters/l2_to_l1.pt",
                 retrieval_json_path="./data/train/raw/retrieval_train.json",
                 cache_path="./data/train/cache/training_samples_cache.pt"):
        
        self.cache_path = cache_path
        self.device = "cpu"
        
        print("1. 正在加载工具字典与映射表...")
        with open(tools_json_path, "r", encoding="utf-8") as f:
            self.tools = json.load(f)
            
        self.tool_to_l2 = torch.load(tool_to_l2_path, weights_only=False)
        self.l2_to_l1 = torch.load(l2_to_l1_path, weights_only=False)
        
        # 构建快速查找字典
        self.id_to_tool = {t["tool_id"]: t for t in self.tools}
        self.name_to_id = {}
        for t in self.tools:
            api_norm = normalize_name(t["api_name"])
            tool_norm = normalize_name(t["tool_name"])
            self.name_to_id[f"{api_norm}for{tool_norm}"] = t["tool_id"]

        # 2. 构建或加载训练样本
        if os.path.exists(self.cache_path):
            print(f"2. 发现缓存文件，直接加载样本: {self.cache_path}")
            self.samples = torch.load(self.cache_path, weights_only=False)
        else:
            print("2. 未发现缓存，正在解析单步检索数据构建训练样本...")
            self.samples = self._build_samples(retrieval_json_path)
            torch.save(self.samples, self.cache_path)
            print(f"   -> 样本已缓存至: {self.cache_path}")

        print(f"✅ 数据集加载完毕！共准备好 {len(self.samples)} 条训练数据。")

    def _build_samples(self, retrieval_json_path):
        if not os.path.exists(retrieval_json_path):
            raise FileNotFoundError(f"找不到文件: {retrieval_json_path}")
            
        with open(retrieval_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = []
        for item in tqdm(data, desc="解析单步检索数据"):
            conversations = item.get("conversations", [])
            query_text = ""
            target_str = ""
            
            # 提取 User 的 Query 和 Assistant 的 Target
            for conv in conversations:
                if conv.get("role") == "user":
                    query_text = conv.get("content", "").strip()
                elif conv.get("role") == "assistant":
                    target_str = conv.get("content", "").strip()
                    
            if not query_text or not target_str:
                continue
                
            # 从 <<Tool&&Api>> 中提取名字
            match = re.match(r'<<(.+?)&&(.+?)>>', target_str)
            if not match:
                continue
                
            tool_name = match.group(1).strip()
            api_name = match.group(2).strip()
            
            # 组装为字典里匹配用的 key
            target_norm = normalize_name(f"{api_name}for{tool_name}")
            
            if target_norm not in self.name_to_id:
                continue # 跳过属于那 15% 隔离区的未知工具
                
            target_id = self.name_to_id[target_norm]

            # 如果这个工具在聚类时被判定为噪声（没有 L2 盒子），直接跳过这条训练数据！
            if target_id not in self.tool_to_l2:
                continue
            
            # 单步检索的核心：永远是冷启动 (-1)
            samples.append({
                "query_text": query_text,
                "t_prev_id": -1,
                "target_tool_id": target_id
            })
                
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        query_text = sample["query_text"]
        t_prev_id = sample["t_prev_id"]
        target_tool_id = sample["target_tool_id"]
        
        
        target_l2 = self.tool_to_l2[target_tool_id]
        target_l1 = self.l2_to_l1[target_l2]
        
        target_tool_info = self.id_to_tool[target_tool_id]
        target_generation_text = f"<<{target_tool_info['tool_name']}&&{target_tool_info['api_name']}>>"

        return {
            "t_prev_id": torch.tensor(t_prev_id, dtype=torch.long),
            "target_l2_id": torch.tensor(target_l2, dtype=torch.long),
            "target_l1_id": torch.tensor(target_l1, dtype=torch.long),
            "tool_label": torch.tensor(target_tool_id, dtype=torch.long),
            "query_text": query_text,
            "target_generation_text": target_generation_text
        }

def get_dataloader(batch_size=4, shuffle=True, num_workers=4):
    dataset = IH_ToolDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader