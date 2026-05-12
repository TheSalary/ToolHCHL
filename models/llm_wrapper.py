import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class SoftPromptPool(nn.Module):
    def __init__(self, num_l2_boxes, k_tokens, hidden_size):
        super().__init__()
        self.k_tokens = k_tokens
        # 为每个 L2 Box 初始化专属的 K 个 Soft Prompts
        self.prompts = nn.Parameter(torch.randn(num_l2_boxes, k_tokens, hidden_size))
        
    def forward(self, l2_box_id):
        # 提取目标 L2 Box 的专属 Soft Prompts，形状为 (batch_size, K, hidden_size)
        return self.prompts[l2_box_id]

class GenerativeToolCaller(nn.Module):
    def __init__(self, model_name_or_path, num_l2_boxes, k_tokens):
        super().__init__()
        # 1. 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        # 2. 加载大模型 (增加 bfloat16 和 device_map)
        print(f"正在以 bfloat16 精度加载本地模型：{model_name_or_path} ... (可能需要几十秒)")
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,  # 关键：开启半精度，否则 4090 显存会爆
            device_map="auto",           # 关键：自动把模型分配到你的 4090 显卡上
            trust_remote_code=True
        )
        
        # 冻结主干 LLM 的所有参数
        for param in self.llm.parameters():
            param.requires_grad = False
            
        self.hidden_size = self.llm.config.hidden_size
        self.prompt_pool = SoftPromptPool(num_l2_boxes, k_tokens, self.hidden_size)
        
    def forward(self, query_texts, l2_box_ids, target_texts=None):
        # 0. 获取大模型所在的设备和精度 (Dtype)
        target_device = self.llm.model.embed_tokens.weight.device
        target_dtype = self.llm.dtype # 拿到模型当前使用的精度，通常是 torch.bfloat16
        
        # 1. 提取专属 Soft Prompts 并对齐设备与精度 
        soft_prompts = self.prompt_pool(l2_box_ids).to(device=target_device, dtype=target_dtype)
        
        # 2. 将 Query 编码为 Word Embeddings [cite: 7]
        query_inputs = self.tokenizer(
            query_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(target_device)
        
        # 获取冻结的 embedding 层的输出
        # 这里的输出通常会自动匹配模型的 dtype，但我们手动再强制转换一次保险
        query_embeds = self.llm.get_input_embeddings()(query_inputs.input_ids).to(target_dtype)
        
        # 3. 序列拼接：[Soft_Prompts, Query_Embeds] 
        inputs_embeds = torch.cat([soft_prompts, query_embeds], dim=1)
        
        # 拼接对应的 attention_mask
        batch_size = inputs_embeds.shape[0]
        prompt_mask = torch.ones(
            (batch_size, self.prompt_pool.k_tokens), 
            dtype=torch.long, 
            device=target_device
        )
        attention_mask = torch.cat([prompt_mask, query_inputs.attention_mask], dim=1)
        
        # 4. 前向传播与任务生成损失 (L_task) [cite: 10, 46]
        if target_texts is not None:
            target_inputs = self.tokenizer(
                target_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            ).to(target_device)
            
            target_embeds = self.llm.get_input_embeddings()(target_inputs.input_ids).to(target_dtype)
            
            # 将目标序列也拼接上去用于计算自回归损失
            full_embeds = torch.cat([inputs_embeds, target_embeds], dim=1)
            full_mask = torch.cat([attention_mask, target_inputs.attention_mask], dim=1)
            
            # 构造 Labels，仅对工具 Token 进行梯度反传 [cite: 46]
            labels = torch.full(full_mask.shape, -100, dtype=torch.long, device=target_device)
            labels[:, -target_inputs.input_ids.shape[1]:] = target_inputs.input_ids
            
            outputs = self.llm(inputs_embeds=full_embeds, attention_mask=full_mask, labels=labels)
            return outputs.loss