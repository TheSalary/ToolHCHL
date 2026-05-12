"""
models/router.py
================
Router 模块统一入口。

├── BaseRouter          — 抽象基类，定义所有 Router 必须实现的接口
├── DualStreamRouter    — 完整双流路由器（默认）
├── FlatDualStreamRouter — 无 L1 层级扁平变体
├── SimpleLinearRouter  — 无 box 机制的极简线性路由器
└── create_router()     — 工厂函数，根据配置创建合适的 Router 实例

使用示例：
    from models.router import create_router, DualStreamRouter

    # 工厂方式（推荐）
    router = create_router(
        router_type="dual_stream",
        dim=384, num_tools=11112, num_boxes=582,
        m_global=m_global_tensor,
        l2_centers=l2_c, l2_widths=l2_w,
        l1_centers=l1_c, l1_widths=l1_w,
    )

    # 直接使用
    router = DualStreamRouter(dim=384, num_tools=11112, num_boxes=582, ...)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional


# ============================================================================
# 1. 抽象基类
# ============================================================================
class BaseRouter(ABC, nn.Module):
    """
    所有 Router 的抽象基类。

    子类必须实现：
        forward(q_vec, t_prev) -> (s_total, s_sem)
            - s_total: [Batch, num_boxes] 融合后的盒子打分
            - s_sem:   [Batch, num_boxes] 纯语义流打分（用于辅助损失）

    子类可选覆盖：
        has_geo_losses  — 是否支持几何包含损失（默认 True）
    """

    @abstractmethod
    def forward(self, q_vec: torch.Tensor, t_prev: torch.Tensor):
        """
        参数:
            q_vec:  [Batch, dim] 用户 Query 向量
            t_prev: [Batch]      上一步调用的工具 ID（-1=冷启动）
        返回:
            (s_total, s_sem): 各 L2 盒子打分元组
        """
        ...

    @property
    def has_geo_losses(self) -> bool:
        """Router 是否包含几何参数（l2_centers/l2_widths），用于损失计算分支。"""
        return True


# ============================================================================
# 2. DualStreamRouter（默认完整路由器）
# ============================================================================
class DualStreamRouter(BaseRouter):
    """
    双流路由器（完整版）。

    语义流：对所有 L2 盒子计算 softmax(-||q - l2||^2)
    依赖流：基于马尔可夫转移矩阵 m_global，从 t_prev 推断下一工具的概率分布
    门控融合：MLP 动态学习 alpha，将语义流和依赖流加权融合
    """

    def __init__(
        self, dim: int, num_tools: int, num_boxes: int,
        m_global: torch.Tensor,
        l2_centers: torch.Tensor, l2_widths: torch.Tensor,
        l1_centers: torch.Tensor, l1_widths: torch.Tensor,
    ):
        super().__init__()
        self.dim = dim
        self.num_tools = num_tools
        self.num_boxes = num_boxes

        # --- 物理空间几何参数（可学习）---
        self.l2_centers = nn.Parameter(l2_centers)
        self.l2_widths  = nn.Parameter(l2_widths)
        self.l1_centers = nn.Parameter(l1_centers)
        self.l1_widths  = nn.Parameter(l1_widths)

        # --- 依赖流先验图（Buffer，随模型保存但不参与梯度更新）---
        self.register_buffer("m_global", m_global)

        # --- 门控网络 ---
        # Embedding 最后一行（索引 = num_tools）是冷启动(-1)的专属占位向量
        self.tool_emb = nn.Embedding(num_tools + 1, dim, padding_idx=num_tools)
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, q_vec, t_prev):
        is_cold_start = (t_prev == -1)

        # Embedding 用索引 num_tools（冷启动占位），m_global 用索引 0
        t_prev_emb = t_prev.clone()
        t_prev_emb[is_cold_start] = self.num_tools
        t_prev_m = t_prev.clone()
        t_prev_m[is_cold_start] = 0

        # --- 语义流 ---
        dist_sq = torch.cdist(q_vec, self.l2_centers, p=2).pow(2)
        dist_sq = torch.clamp(dist_sq, max=50.0)
        s_sem = F.softmax(-dist_sq, dim=-1)

        # --- 依赖流 ---
        s_dep = self.m_global[t_prev_m]
        s_dep[is_cold_start] = 0.0

        # --- 门控融合 ---
        t_emb = self.tool_emb(t_prev_emb)
        gate_input = torch.cat([q_vec, t_emb], dim=-1)
        alpha = self.gate_mlp(gate_input)

        is_cold_mask = is_cold_start.unsqueeze(-1)
        alpha_safe = torch.where(is_cold_mask, torch.ones_like(alpha), alpha)

        # 维度自适应：依赖流可能比语义流短（增量学习场景），补零对齐
        if s_dep.shape[1] != s_sem.shape[1]:
            padding = torch.zeros(
                (s_dep.shape[0], s_sem.shape[1] - s_dep.shape[1]),
                device=s_dep.device, dtype=s_dep.dtype,
            )
            s_dep = torch.cat([s_dep, padding], dim=1)

        s_total = alpha_safe * s_sem + (1.0 - alpha_safe) * s_dep
        return s_total, s_sem


# ============================================================================
# 3. FlatDualStreamRouter（L1 层级消融变体）
# ============================================================================
class FlatDualStreamRouter(BaseRouter):
    """
    扁平化双流路由器（L1 层级消融专用变体）。

    与 DualStreamRouter 的核心区别：
    - 语义流相同，直接对 L2 盒子打分
    - 依赖流基于工具级转移矩阵 m_global_tool（替代 L1 层级加权聚合）
    - 不使用 L1 中心/宽度，完全绕过 L1 层的加权聚合逻辑
    """

    def __init__(
        self, dim: int, num_tools: int, num_boxes: int,
        m_global_tool: torch.Tensor,
        l2_centers: torch.Tensor, l2_widths: torch.Tensor,
        l1_centers=None, l1_widths=None,  # 保留参数签名兼容，但不使用
    ):
        super().__init__()
        self.dim = dim
        self.num_tools = num_tools
        self.num_boxes = num_boxes

        self.l2_centers = nn.Parameter(l2_centers)
        self.l2_widths  = nn.Parameter(l2_widths)
        self.register_buffer("m_global_tool", m_global_tool)

        self.tool_emb = nn.Embedding(num_tools + 1, dim, padding_idx=num_tools)
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    @property
    def has_geo_losses(self) -> bool:
        return True  # 有 l2_centers/l2_widths，支持几何损失

    def forward(self, q_vec, t_prev):
        is_cold_start = (t_prev == -1)

        # --- 语义流（与 DualStreamRouter 相同）---
        dist_sq = torch.cdist(q_vec, self.l2_centers, p=2).pow(2)
        dist_sq = torch.clamp(dist_sq, max=50.0)
        s_sem = F.softmax(-dist_sq, dim=-1)

        # --- 依赖流：工具级矩阵查表（替代 L1 加权聚合）---
        t_prev_m = t_prev.clone()
        t_prev_m[is_cold_start] = 0
        s_dep = self.m_global_tool[t_prev_m]
        s_dep[is_cold_start] = 0.0

        # --- 门控融合 ---
        t_prev_emb = t_prev.clone()
        t_prev_emb[is_cold_start] = self.num_tools
        t_emb = self.tool_emb(t_prev_emb)
        gate_input = torch.cat([q_vec, t_emb], dim=-1)
        alpha = self.gate_mlp(gate_input)

        is_cold_mask = is_cold_start.unsqueeze(-1)
        alpha_safe = torch.where(is_cold_mask, torch.ones_like(alpha), alpha)

        s_total = alpha_safe * s_sem + (1.0 - alpha_safe) * s_dep
        return s_total, s_sem


# ============================================================================
# 4. SimpleLinearRouter（无 box 机制的极简路由器）
# ============================================================================
class SimpleLinearRouter(BaseRouter):
    """
    极简线性路由器（无 box 机制）。

    与 DualStreamRouter 的核心区别：
    - 去掉所有几何参数（l2_centers/l2_widths/l1_centers/l1_widths）
    - 去掉 m_global 依赖矩阵
    - 用一个线性投影直接从 query 预测 prompt 选择

    用途：消融实验 —— 证明 Box 机制对性能提升有效。
    """

    def __init__(
        self, dim: int, num_tools: int, num_boxes: int,
        l2_centers=None, l2_widths=None, l1_centers=None, l1_widths=None,
        m_global=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_tools = num_tools
        self.num_boxes = num_boxes

        self.prompt_proj = nn.Linear(dim, num_boxes)

    @property
    def has_geo_losses(self) -> bool:
        return False  # 无几何参数

    def forward(self, q_vec, t_prev=None):
        target_dtype = next(self.prompt_proj.parameters()).dtype
        if q_vec.dtype != target_dtype:
            q_vec = q_vec.to(target_dtype)

        raw_scores = self.prompt_proj(q_vec)
        prompt_probs = F.softmax(raw_scores, dim=-1)
        return prompt_probs, prompt_probs


# 保留旧别名，确保向后兼容
LinearBoxRouter = SimpleLinearRouter


# ============================================================================
# 5. 工厂函数
# ============================================================================
# Router 类型常量（供 create_router 和其他模块使用）
ROUTER_TYPE_DUAL_STREAM   = "dual_stream"
ROUTER_TYPE_FLAT          = "flat"
ROUTER_TYPE_LINEAR         = "linear"
ROUTER_TYPE_SIMPLE_LINEAR  = "simple_linear"   # SimpleLinearRouter 的别名

# Ablation 名称到 Router 类型的映射
ABLATION_TO_ROUTER_TYPE = {
    "semi_freeze_off":    ROUTER_TYPE_DUAL_STREAM,
    "weight_inherit_off": ROUTER_TYPE_DUAL_STREAM,
    "replay_off":         ROUTER_TYPE_DUAL_STREAM,
    "geo_loss_off":       ROUTER_TYPE_DUAL_STREAM,
    "w/o_hierarchy":      ROUTER_TYPE_FLAT,
    "flat_space":         ROUTER_TYPE_FLAT,
    "linear_router":      ROUTER_TYPE_SIMPLE_LINEAR,
    # 推理时消融使用基线路由器，动态包装
    "router_semantic":    ROUTER_TYPE_DUAL_STREAM,
    "router_dependency":  ROUTER_TYPE_DUAL_STREAM,
    "router_no_gate":      ROUTER_TYPE_DUAL_STREAM,
}


def create_router(
    router_type: str,
    dim: int, num_tools: int, num_boxes: int,
    m_global: Optional[torch.Tensor] = None,
    l2_centers: Optional[torch.Tensor] = None,
    l2_widths:  Optional[torch.Tensor] = None,
    l1_centers: Optional[torch.Tensor] = None,
    l1_widths:  Optional[torch.Tensor] = None,
    m_global_tool: Optional[torch.Tensor] = None,
) -> BaseRouter:
    """
    工厂函数：根据 router_type 创建合适的 Router 实例。

    参数:
        router_type: "dual_stream" | "flat" | "linear" | "simple_linear"
        dim, num_tools, num_boxes: 基础维度参数
        m_global:  DualStreamRouter 依赖流矩阵 [num_tools, num_boxes]
        l2_centers / l2_widths: L2 盒子几何参数 [num_boxes, dim]
        l1_centers / l1_widths: L1 盒子几何参数 [num_l1, dim]
        m_global_tool: FlatDualStreamRouter 工具级依赖矩阵 [num_tools, num_boxes]

    返回:
        BaseRouter 子类实例

    示例:
        router = create_router(
            router_type="dual_stream",
            dim=384, num_tools=11112, num_boxes=582,
            m_global=m_global, l2_centers=l2_c, l2_widths=l2_w,
            l1_centers=l1_c, l1_widths=l1_w,
        )
    """
    if router_type in (ROUTER_TYPE_DUAL_STREAM, "dual"):
        if m_global is None:
            raise ValueError("create_router(dual_stream): 需要传入 m_global")
        if l2_centers is None or l2_widths is None:
            raise ValueError("create_router(dual_stream): 需要传入 l2_centers 和 l2_widths")
        return DualStreamRouter(
            dim=dim, num_tools=num_tools, num_boxes=num_boxes,
            m_global=m_global,
            l2_centers=l2_centers, l2_widths=l2_widths,
            l1_centers=l1_centers, l1_widths=l1_widths,
        )

    elif router_type in (ROUTER_TYPE_FLAT, "flat_dual"):
        if m_global_tool is None:
            raise ValueError("create_router(flat): 需要传入 m_global_tool")
        return FlatDualStreamRouter(
            dim=dim, num_tools=num_tools, num_boxes=num_boxes,
            m_global_tool=m_global_tool,
            l2_centers=l2_centers, l2_widths=l2_widths,
        )

    elif router_type in (ROUTER_TYPE_LINEAR, ROUTER_TYPE_SIMPLE_LINEAR, "linear_box"):
        return SimpleLinearRouter(
            dim=dim, num_tools=num_tools, num_boxes=num_boxes,
        )

    else:
        raise ValueError(
            f"未知 router_type={router_type!r}，"
            f"可选: 'dual_stream' | 'flat' | 'linear'"
        )
