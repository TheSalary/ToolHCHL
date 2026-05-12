import torch
import torch.nn as nn
import torch.nn.functional as F

class IHLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def compute_geo_loss(self, l1_c, l1_w, l2_c, l2_w):
        """
        计算几何包含损失 (Geometric Inclusion Loss)
        目标：严格约束 L2 盒子必须被包含在 L1 物理墙内。
        如果 L2 越界，则产生极大的惩罚 (Penalty)。
        """
        # L2 的边界：c - w 和 c + w
        # L1 的边界：c - w 和 c + w
        
        # 左侧越界量 (L2的左边界 比 L1的左边界 还要靠左)
        out_left = F.relu((l1_c - l1_w) - (l2_c - l2_w))
        # 右侧越界量 (L2的右边界 比 L1的右边界 还要靠右)
        out_right = F.relu((l2_c + l2_w) - (l1_c + l1_w))
        
        # 越界量之和作为 Loss
        loss_geo = torch.mean(out_left + out_right)
        return loss_geo

    def compute_contrastive_loss(self, s_total, target_l2):
        """
        计算路由对比损失 (Contrastive Routing Loss)
        s_total: Router 输出的概率分布 [Batch, num_boxes]
        target_l2: 真实的目标盒子 ID [Batch]
        """
        # s_total 已经是概率分布 (Softmax 之后的或者门控融合后的)
        # clamp 防止 log(0) -> -inf -> NaN
        s_total = torch.clamp(s_total, min=1e-7, max=1.0)
        log_probs = torch.log(s_total)
        
        # 使用负对数似然损失 (NLLLoss)
        loss_cont = F.nll_loss(log_probs, target_l2)
        return loss_cont