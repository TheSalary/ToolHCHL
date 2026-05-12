"""
training/ablation_components.py
===============================
消融实验专用组件（AblationRouter + AblationLoss）。

从 run_ablation.py 提取并重构，依赖新的 models/router.py 基类体系。

使用方式
--------
    from training.ablation_components import AblationRouter, AblationLoss

    # 推理时消融（无需重新训练）
    base_router = create_router("dual_stream", ...)
    router = AblationRouter(base_router, ablation_mode="router_semantic")

    # 训练时消融（需要重新训练）
    loss_fn = AblationLoss(IHLoss(), geo_off=True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ============================================================================
# 1. AblationRouter — 推理时消融路由器
# ============================================================================
class AblationRouter(nn.Module):
    """
    支持推理时消融的路由模块（包装 BaseRouter 子类）。

    ablation_mode:
        - None / "full":        完整双流路由（直接透传 base_router）
        - "router_semantic":    仅语义流（去除依赖流和门控）
        - "router_dependency":  仅依赖流（冷启动退回到语义流）
        - "router_no_gate":     去除门控，固定 alpha=0.5 融合
    """

    def __init__(self, base_router, ablation_mode: Optional[str] = None):
        super().__init__()
        self.base_router = base_router
        self.ablation_mode = ablation_mode or "full"

    @property
    def has_geo_losses(self) -> bool:
        return self.base_router.has_geo_losses

    def load_state_dict(self, state_dict, strict=True):
        self.base_router.load_state_dict(state_dict, strict=strict)

    def state_dict(self, *args, **kwargs):
        return self.base_router.state_dict(*args, **kwargs)

    def _pad_dep(self, s_dep, target_len):
        """将依赖流 padding 到与语义流相同的 num_boxes 长度。"""
        if s_dep.shape[1] < target_len:
            padding = torch.zeros(
                s_dep.shape[0], target_len - s_dep.shape[1],
                device=s_dep.device, dtype=s_dep.dtype,
            )
            return torch.cat([s_dep, padding], dim=1)
        return s_dep[:, :target_len]

    def forward(self, q_vec, t_prev):
        """
        参数:
            q_vec:  [Batch, dim]
            t_prev: [Batch] 上一步工具 ID（-1=冷启动）
        返回:
            (s_total, s_sem): 盒子打分元组
        """
        mode = self.ablation_mode

        # 所有模式都计算语义流（门控网络依赖 base_router 的组件）
        # 先调用 base_router 的内部计算逻辑
        # base_router.forward 计算 s_sem
        is_cold_start = (t_prev == -1)
        num_tools = self.base_router.num_tools

        # 重新执行 base_router 的前几步（获取 s_sem）
        # 注意：base_router.forward 会内部计算所有流并融合，
        # 这里我们只取其中的 s_sem 部分，自己做融合控制

        # --- 语义流（所有模式都需要）---
        l2_centers = self.base_router.l2_centers
        dist_sq = torch.cdist(q_vec, l2_centers, p=2).pow(2)
        dist_sq = torch.clamp(dist_sq, max=50.0)
        s_sem = F.softmax(-dist_sq, dim=-1)

        # 透传模式：直接调用 base_router
        if mode == "full" or mode is None:
            return self.base_router(q_vec, t_prev)

        # --- 仅语义流 ---
        if mode == "router_semantic":
            return s_sem, s_sem

        # --- 依赖流 ---
        m_global = self.base_router.m_global
        t_prev_m = t_prev.clone()
        t_prev_m[is_cold_start] = 0
        s_dep = m_global[t_prev_m]
        s_dep[is_cold_start] = 0.0

        # --- 仅依赖流（冷启动退回到语义流）---
        if mode == "router_dependency":
            s_dep = self._pad_dep(s_dep, s_sem.shape[1])
            is_cold_expanded = is_cold_start.unsqueeze(-1)
            s_total = torch.where(is_cold_expanded, s_sem, s_dep)
            return s_total, s_sem

        # --- 去除门控，固定 alpha=0.5 ---
        if mode == "router_no_gate":
            s_dep = self._pad_dep(s_dep, s_sem.shape[1])
            alpha = 0.5
            is_cs = is_cold_start.unsqueeze(-1)
            alpha_safe = torch.where(is_cs, torch.ones_like(s_sem[:, :1]), torch.full_like(s_sem[:, :1], 0.5))
            s_total = alpha_safe * s_sem + (1.0 - alpha_safe) * s_dep
            return s_total, s_sem

        # 默认：透传
        return self.base_router(q_vec, t_prev)


# ============================================================================
# 2. AblationLoss — 消融损失函数
# ============================================================================
class AblationLoss(nn.Module):
    """
    支持关闭特定损失项的损失计算器。

    包装 IHLoss，根据配置选择性地禁用：
    - geo_off:      关闭几何包含损失
    - contrast_off: 关闭路由对比损失
    """

    def __init__(self, base_loss_fn, geo_off: bool = False, contrast_off: bool = False):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.geo_off = geo_off
        self.contrast_off = contrast_off

    def compute_geo_loss(self, l1_c, l1_w, l2_c, l2_w):
        if self.geo_off or l1_c is None or l2_c is None:
            device = l2_c.device if l2_c is not None else (l1_c.device if l1_c is not None else "cpu")
            return torch.tensor(0.0, device=device)
        return self.base_loss_fn.compute_geo_loss(l1_c, l1_w, l2_c, l2_w)

    def compute_contrastive_loss(self, s_total, target_l2):
        if self.contrast_off:
            return torch.tensor(0.0, device=s_total.device)
        return self.base_loss_fn.compute_contrastive_loss(s_total, target_l2)
