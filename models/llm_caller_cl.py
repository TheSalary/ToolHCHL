"""
models/llm_caller_cl.py
=======================
向后兼容模块。

所有内容已迁移到 models/llm_caller.py，此文件仅作为别名导出，
确保现有代码（如 run_ablation.py 的 `from models.llm_caller_cl import LLMCaller as CLCaller`）
无需修改即可正常工作。

新代码推荐直接使用：
    from models.llm_caller import LLMCallerCL, create_llm_caller
"""

# 向后兼容：直接引用 llm_caller.py 中对应的类
from models.llm_caller import (
    LLMCallerCL,
    NoPromptLLMCaller,
    create_llm_caller,
    CALLER_TYPE_BASE,
    CALLER_TYPE_CL,
    CALLER_TYPE_NO_PROMPT,
)

# 保留原始类名，方便现有代码直接替换 import
LLMCaller = LLMCallerCL  # 等价于原 llm_caller_cl.py 中的 LLMCaller（CL版）

__all__ = [
    "LLMCallerCL",
    "LLMCaller",         # 别名
    "NoPromptLLMCaller",
    "create_llm_caller",
    "CALLER_TYPE_BASE",
    "CALLER_TYPE_CL",
    "CALLER_TYPE_NO_PROMPT",
]
