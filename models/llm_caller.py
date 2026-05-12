"""
models/llm_caller.py
====================
LLMCaller 模块统一入口，提供统一继承体系和工厂函数。

类层次结构
----------
    BaseLLMCaller (ABC)
        抽象基类，定义所有 LLMCaller 必须实现的接口。
        ├── extract_query_vector(text_list) -> (q_vec, input_ids, attention_mask)
        ├── forward(input_ids, attention_mask, box_ids) -> logits
        └── 属性: num_boxes, num_tools, has_prompt_pool

    BaseLLMCallerImpl (nn.Module)
        通用实现：extract_query_vector（embedding-layer 路径）
        ├── LLMCallerBase — 基线版（单一 prompt_pool + 单一 classifier）
        └── LLMCallerCL   — CL 版（old/new 分离的 prompt_pool + classifier）

工厂函数
--------
    create_llm_caller() — 根据类型创建合适的实例

使用示例
--------
    from models.llm_caller import create_llm_caller, LLMCallerBase, LLMCallerCL

    # 基线版
    caller = create_llm_caller(
        caller_type="base",
        base_llm=llm, tokenizer=tokenizer,
        num_boxes=582, num_tools=11112,
    )

    # CL 版
    caller = create_llm_caller(
        caller_type="cl",
        base_llm=llm, tokenizer=tokenizer,
        num_boxes=632, num_tools=11752,
        old_num_boxes=582, old_num_tools=11112,
    )
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


# ============================================================================
# 1. 抽象基类
# ============================================================================
class BaseLLMCaller(ABC, nn.Module):
    """
    所有 LLMCaller 的抽象基类。

    子类必须实现：
        extract_query_vector(text_list) -> (q_vec, input_ids, attention_mask)
        forward(input_ids, attention_mask, box_ids) -> logits
            - logits: [Batch, num_tools] 工具分类 logits
    """

    @property
    @abstractmethod
    def num_boxes(self) -> int:
        """总盒子数（含 old + new）。"""
        ...

    @property
    @abstractmethod
    def num_tools(self) -> int:
        """总工具数（含 old + new）。"""
        ...

    @property
    def has_prompt_pool(self) -> bool:
        """是否有 Prompt Pool（有则需要 box_id 来索引）。"""
        return True

    @abstractmethod
    def extract_query_vector(self, text_list):
        """
        从文本列表提取 Query 向量。

        返回:
            q_vec:        [Batch, router_dim] 投影后的 query 向量
            input_ids:    [Batch, seq_len] tokenized input
            attention_mask: [Batch, seq_len] attention mask
        """
        ...

    @abstractmethod
    def forward(self, input_ids, attention_mask, box_ids):
        """
        推理正向传播。

        参数:
            input_ids:     [Batch, seq_len]
            attention_mask: [Batch, seq_len]
            box_ids:       [Batch] 各样本对应的 L2 盒子 ID
        返回:
            logits: [Batch, num_tools]
        """
        ...


# ============================================================================
# 2. 通用实现混入类（提供 extract_query_vector）
# ============================================================================
class BaseLLMCallerImpl(BaseLLMCaller):
    """
    通用实现混入类（Mixins 风格）。

    提供 extract_query_vector 的两种路径：
    - embedding_only=True：仅用 embedding layer（avg pooling），不跑 transformer
      （适用于 Base 训练，节省显存）
    - embedding_only=False：跑 transformer 前几层 + avg pooling
      （适用于 CL 训练，利用更多语义信息）
    """

    def __init__(self, base_llm, tokenizer, router_dim=384, prompt_length=10,
                 embedding_only=True, embedding_layer_idx=1):
        super().__init__()
        self.base_llm = base_llm
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.router_dim = router_dim
        self.embedding_only = embedding_only
        self.embedding_layer_idx = embedding_layer_idx

        hidden_dim = base_llm.config.hidden_size

        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim, 1024, dtype=base_llm.dtype),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(1024, router_dim, dtype=base_llm.dtype),
        )

        if embedding_only:
            self.ln = nn.LayerNorm(hidden_dim, dtype=base_llm.dtype)

    def _extract_embedding_only(self, text_list):
        """路径 A：仅用 embedding layer（Base 训练时使用）。"""
        device = next(self.parameters()).device
        inputs = self.tokenizer(text_list, return_tensors="pt",
                                padding=True, truncation=True).to(device)

        with torch.no_grad():
            embeddings = self.base_llm.model.embed_tokens(inputs.input_ids)
            embeddings = self.ln(embeddings)

            mask_expanded = inputs.attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_hidden = torch.sum(embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            q_vec_avg = sum_hidden / sum_mask

        q_vec_avg = q_vec_avg.to(self.base_llm.dtype)
        q_vec_projected = self.query_proj(q_vec_avg)
        return q_vec_projected, inputs.input_ids, inputs.attention_mask

    def _extract_transformer_layers(self, text_list):
        """路径 B：跑 transformer 前几层（CL 训练时使用）。"""
        device = next(self.parameters()).device
        inputs = self.tokenizer(text_list, return_tensors="pt",
                                padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = self.base_llm(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            layer_hidden = outputs.hidden_states[self.embedding_layer_idx]
            mask_expanded = inputs.attention_mask.unsqueeze(-1).expand(layer_hidden.size()).float()
            sum_hidden = torch.sum(layer_hidden * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            q_vec_avg = sum_hidden / sum_mask

        q_vec_avg = q_vec_avg.to(self.base_llm.dtype)
        q_vec_projected = self.query_proj(q_vec_avg)
        return q_vec_projected, inputs.input_ids, inputs.attention_mask

    def extract_query_vector(self, text_list):
        if self.embedding_only:
            return self._extract_embedding_only(text_list)
        else:
            return self._extract_transformer_layers(text_list)


# ============================================================================
# 3. LLMCallerBase — 基线版（单 prompt_pool + 单 classifier）
# ============================================================================
class LLMCallerBase(BaseLLMCallerImpl):
    """
    基线版 LLMCaller。

    单一 prompt_pool + 单一 classifier，适用于 Base 训练和评估。
    """

    def __init__(self, base_llm, tokenizer, num_boxes, num_tools,
                 router_dim=384, prompt_length=10):
        super().__init__(
            base_llm=base_llm, tokenizer=tokenizer,
            router_dim=router_dim, prompt_length=prompt_length,
            embedding_only=True,  # Base 用 embedding-only 节省显存
        )
        hidden_dim = base_llm.config.hidden_size
        self._num_boxes = num_boxes
        self._num_tools = num_tools

        self.prompt_pool = nn.Parameter(
            torch.randn(num_boxes, prompt_length, hidden_dim,
                        dtype=base_llm.dtype) * 0.02,
        )
        self.classifier = nn.Linear(hidden_dim, num_tools, dtype=base_llm.dtype)

    @property
    def num_boxes(self) -> int:
        return self._num_boxes

    @property
    def num_tools(self) -> int:
        return self._num_tools

    def forward(self, input_ids, attention_mask, box_ids):
        selected_soft_prompts = self.prompt_pool[box_ids]

        inputs_embeds = self.base_llm.model.embed_tokens(input_ids)
        full_inputs_embeds = torch.cat([selected_soft_prompts, inputs_embeds], dim=1)

        batch_size = input_ids.shape[0]
        prompt_mask = torch.ones(
            (batch_size, self.prompt_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        outputs = self.base_llm(
            inputs_embeds=full_inputs_embeds,
            attention_mask=full_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_layer_hidden = outputs.hidden_states[-1]
        logits = self.classifier(last_layer_hidden[:, -1, :])
        return logits


# ============================================================================
# 4. LLMCallerCL — CL 版（old/new 分离）
# ============================================================================
class LLMCallerCL(BaseLLMCallerImpl):
    """
    持续学习版 LLMCaller。

    prompt_pool 和 classifier 均物理切割为 old/new 两部分：
    - prompt_pool_old：冻结，保存旧任务记忆
    - prompt_pool_new：可学习，适配新任务
    - classifier_old：冻结，防止遗忘
    - classifier_new：可学习，适配新工具集
    """

    def __init__(self, base_llm, tokenizer, num_boxes, num_tools,
                 old_num_boxes, old_num_tools,
                 router_dim=384, prompt_length=10):
        super().__init__(
            base_llm=base_llm, tokenizer=tokenizer,
            router_dim=router_dim, prompt_length=prompt_length,
            embedding_only=False,  # CL 用 transformer 层获得更多语义
        )
        self._num_boxes = num_boxes
        self._num_tools = num_tools
        self.old_num_boxes = old_num_boxes
        self.old_num_tools = old_num_tools
        self.base_llm.gradient_checkpointing_enable()

        hidden_dim = base_llm.config.hidden_size
        new_boxes = num_boxes - old_num_boxes
        new_tools = num_tools - old_num_tools

        # prompt_pool：old 冻结，new 可学习
        self.prompt_pool_old = nn.Parameter(
            torch.zeros(old_num_boxes, prompt_length, hidden_dim,
                        dtype=base_llm.dtype),
            requires_grad=False,
        )
        self.prompt_pool_new = nn.Parameter(
            torch.randn(new_boxes, prompt_length, hidden_dim,
                        dtype=base_llm.dtype) * 0.02,
            requires_grad=True,
        )

        # classifier：old 冻结，new 可学习
        self.classifier_old = nn.Linear(hidden_dim, old_num_tools,
                                        dtype=base_llm.dtype)
        self.classifier_old.weight.requires_grad = False
        if self.classifier_old.bias is not None:
            self.classifier_old.bias.requires_grad = False

        self.classifier_new = nn.Linear(hidden_dim, new_tools,
                                       dtype=base_llm.dtype)

    @property
    def num_boxes(self) -> int:
        return self._num_boxes

    @property
    def num_tools(self) -> int:
        return self._num_tools

    @property
    def has_prompt_pool(self) -> bool:
        return True

    def load_base_weights(self, old_prompt_tensor, old_classifier_state_dict):
        """
        从 checkpoint 注入旧任务记忆。

        参数:
            old_prompt_tensor:  [old_num_boxes, prompt_length, hidden_dim]
            old_classifier_state_dict: classifier_old 的权重 state_dict，
                                      或 {'weight_old', 'bias_old'} 格式
        """
        self.prompt_pool_old.data.copy_(old_prompt_tensor.to(self.base_llm.dtype))
        if isinstance(old_classifier_state_dict, dict) and 'weight_old' in old_classifier_state_dict:
            self.classifier_old.weight.data.copy_(
                old_classifier_state_dict['weight_old'].to(self.base_llm.dtype))
            self.classifier_old.bias.data.copy_(
                old_classifier_state_dict['bias_old'].to(self.base_llm.dtype))
        else:
            self.classifier_old.load_state_dict(old_classifier_state_dict)

    def forward(self, input_ids, attention_mask, box_ids):
        batch_size = box_ids.shape[0]
        hidden_dim = self.base_llm.config.hidden_size

        selected_soft_prompts = torch.zeros(
            (batch_size, self.prompt_length, hidden_dim),
            dtype=self.base_llm.dtype,
            device=box_ids.device,
        )

        is_old = box_ids < self.old_num_boxes
        is_new = box_ids >= self.old_num_boxes

        if is_old.any():
            selected_soft_prompts[is_old] = self.prompt_pool_old[box_ids[is_old]]
        if is_new.any():
            selected_soft_prompts[is_new] = self.prompt_pool_new[
                box_ids[is_new] - self.old_num_boxes
            ]

        # 微小梯度干扰，防止全老工具 batch 梯度断流
        selected_soft_prompts = selected_soft_prompts + self.prompt_pool_new.mean() * 0.0

        inputs_embeds = self.base_llm.model.embed_tokens(input_ids)
        full_inputs_embeds = torch.cat([selected_soft_prompts, inputs_embeds], dim=1)

        prompt_mask = torch.ones(
            (batch_size, self.prompt_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        outputs = self.base_llm(
            inputs_embeds=full_inputs_embeds,
            attention_mask=full_attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        last_layer_hidden = outputs.hidden_states[-1]

        logits_old = self.classifier_old(last_layer_hidden[:, -1, :])
        logits_new = self.classifier_new(last_layer_hidden[:, -1, :])
        logits = torch.cat([logits_old, logits_new], dim=-1)
        return logits


# ============================================================================
# 5. NoPromptLLMCaller — 无 Prompt Pool（消融专用）
# ============================================================================
class NoPromptLLMCaller(nn.Module):
    """
    无 Prompt Pool 版本（w/o_prompt_pool 消融专用）。

    冻结 LLM backbone，直接用 LLM hidden states 做 router + classifier。
    证明 Prompt 隔离机制对防止遗忘有效。

    注意：此类不继承 BaseLLMCaller（语义不同：无 prompt pool），
    独立存在。
    """

    def __init__(self, base_llm, tokenizer, num_boxes, num_tools,
                 router_dim=384, old_num_boxes=0, old_num_tools=0):
        super().__init__()
        self.base_llm = base_llm
        self.tokenizer = tokenizer
        self.num_boxes = num_boxes
        self.num_tools = num_tools
        self.old_num_boxes = old_num_boxes
        self.old_num_tools = old_num_tools

        for param in self.base_llm.parameters():
            param.requires_grad = False

        hidden_dim = base_llm.config.hidden_size
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim, 1024, dtype=base_llm.dtype),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(1024, router_dim, dtype=base_llm.dtype),
        )

        if old_num_tools > 0:
            self.classifier_old = nn.Linear(hidden_dim, old_num_tools,
                                           dtype=base_llm.dtype)
            self.classifier_old.weight.requires_grad = False
            if self.classifier_old.bias is not None:
                self.classifier_old.bias.requires_grad = False
            new_tools = num_tools - old_num_tools
            self.classifier_new = nn.Linear(hidden_dim, new_tools,
                                           dtype=base_llm.dtype)
        else:
            self.classifier = nn.Linear(hidden_dim, num_tools, dtype=base_llm.dtype)

    @property
    def has_prompt_pool(self) -> bool:
        return False

    def load_base_weights(self, old_prompt_tensor=None, old_classifier_state_dict=None):
        if old_classifier_state_dict is not None and hasattr(self, 'classifier_old'):
            self.classifier_old.load_state_dict(old_classifier_state_dict)

    def extract_query_vector(self, text_list):
        device = next(self.parameters()).device
        inputs = self.tokenizer(text_list, return_tensors="pt",
                                padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = self.base_llm(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            layer_1_hidden = outputs.hidden_states[1]
            mask_expanded = inputs.attention_mask.unsqueeze(-1).expand(layer_1_hidden.size()).float()
            sum_hidden = torch.sum(layer_1_hidden * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            q_vec_avg = sum_hidden / sum_mask
        q_vec_avg = q_vec_avg.to(self.base_llm.dtype)
        q_vec_projected = self.query_proj(q_vec_avg)
        return q_vec_projected, inputs.input_ids, inputs.attention_mask

    def forward(self, input_ids, attention_mask, box_ids):
        del box_ids  # unused
        with torch.no_grad():
            outputs = self.base_llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1][:, -1, :]

        if hasattr(self, 'classifier'):
            return self.classifier(last_hidden)

        logits_old = self.classifier_old(last_hidden)
        logits_new = self.classifier_new(last_hidden)
        return torch.cat([logits_old, logits_new], dim=-1)


# ============================================================================
# 6. 工厂函数
# ============================================================================
# Caller 类型常量
CALLER_TYPE_BASE = "base"
CALLER_TYPE_CL   = "cl"
CALLER_TYPE_NO_PROMPT = "no_prompt"

# 向后兼容别名：原 run.py 的 _import_base_modules() 依赖这个名字
LLMCaller = LLMCallerBase


def create_llm_caller(
    caller_type: str,
    base_llm, tokenizer,
    num_boxes: int, num_tools: int,
    router_dim: int = 384,
    prompt_length: int = 10,
    old_num_boxes: int = 0,
    old_num_tools: int = 0,
) -> BaseLLMCaller:
    """
    工厂函数：根据 caller_type 创建合适的 LLMCaller 实例。

    参数:
        caller_type: "base" | "cl" | "no_prompt"
        base_llm / tokenizer: LLM 模型和分词器
        num_boxes / num_tools: 总盒子数和总工具数
        router_dim / prompt_length: 向量维度和提示词长度
        old_num_boxes / old_num_tools: CL 版需要指定旧任务边界

    示例:
        caller = create_llm_caller(
            caller_type="cl",
            base_llm=llm, tokenizer=tokenizer,
            num_boxes=632, num_tools=11752,
            old_num_boxes=582, old_num_tools=11112,
        )
    """
    if caller_type == CALLER_TYPE_BASE:
        return LLMCallerBase(
            base_llm=base_llm, tokenizer=tokenizer,
            num_boxes=num_boxes, num_tools=num_tools,
            router_dim=router_dim, prompt_length=prompt_length,
        )

    elif caller_type == CALLER_TYPE_CL:
        if old_num_boxes <= 0 or old_num_tools <= 0:
            raise ValueError(
                f"create_llm_caller(cl): 需要指定 old_num_boxes 和 old_num_tools，"
                f"got old_num_boxes={old_num_boxes}, old_num_tools={old_num_tools}"
            )
        return LLMCallerCL(
            base_llm=base_llm, tokenizer=tokenizer,
            num_boxes=num_boxes, num_tools=num_tools,
            old_num_boxes=old_num_boxes, old_num_tools=old_num_tools,
            router_dim=router_dim, prompt_length=prompt_length,
        )

    elif caller_type == CALLER_TYPE_NO_PROMPT:
        return NoPromptLLMCaller(
            base_llm=base_llm, tokenizer=tokenizer,
            num_boxes=num_boxes, num_tools=num_tools,
            router_dim=router_dim,
            old_num_boxes=old_num_boxes, old_num_tools=old_num_tools,
        )

    else:
        raise ValueError(
            f"未知 caller_type={caller_type!r}，"
            f"可选: 'base' | 'cl' | 'no_prompt'"
        )


# 兼容旧代码的别名
LLMCaller = LLMCallerBase
