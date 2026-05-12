import torch
import torch.nn as nn
import torch.nn.functional as F

class BoxEmbeddingSpace(nn.Module):
    def __init__(self, num_boxes, dim, epsilon=1e-4):
        super().__init__()
        self.dim = dim
        self.epsilon = epsilon
        self.centers = nn.Parameter(torch.randn(num_boxes, dim))
        self.widths = nn.Parameter(torch.ones(num_boxes, dim) * 0.1)
        
    def init_from_clusters(self, cluster_data_dict):
        """采用包围盒策略进行中心点和宽度的初始化 [cite: 18]"""
        with torch.no_grad():
            for box_id, vectors in cluster_data_dict.items():
                max_s = torch.max(vectors, dim=0)[0]
                min_s = torch.min(vectors, dim=0)[0]
                # $$c_{init} = \frac{\max(S) + \min(S)}{2}$$ [cite: 20]
                self.centers[box_id] = (max_s + min_s) / 2.0
                # $$w_{init} = \frac{\max(S) - \min(S)}{2} + \epsilon$$ [cite: 21]
                self.widths[box_id] = (max_s - min_s) / 2.0 + self.epsilon

    def compute_semantic_distance(self, q):
        """
        计算几何惩罚距离。距离越近，激活分数越高 [cite: 26]。
        $$S_{sem}(q, Box_k) = - \sum_{i=1}^d \text{Softplus}(|q_i - c_{k,i}| - w_{k,i})$$ [cite: 27]
        """
        q_exp = q.unsqueeze(1) 
        c_exp = self.centers.unsqueeze(0)
        w_exp = self.widths.unsqueeze(0)
        
        abs_diff = torch.abs(q_exp - c_exp)
        penalty = F.softplus(abs_diff - w_exp)
        return -torch.sum(penalty, dim=-1)