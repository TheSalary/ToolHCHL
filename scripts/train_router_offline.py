import sys
import os
# 把项目根目录加入 Python 路径，这样 scripts/ 子目录运行也能 import models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# 导入你自己的 Router
from models.router import DualStreamRouter

# 🚀 降维打击的核心：Batch Size 直接开到 2048！
BATCH_SIZE = 2048
EPOCHS = 30
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_TOOLS = 11112

# ==========================================
# 📂 全新规划的清爽路径配置
# ==========================================
FEATURES_PATH = "./data/train/features/offline_features.pt"

L2_CENTERS_PATH = "./data/train/clusters/l2_centers.pt"
L2_WIDTHS_PATH = "./data/train/clusters/l2_widths.pt"
L1_CENTERS_PATH = "./data/train/clusters/l1_centers.pt"
L1_WIDTHS_PATH = "./data/train/clusters/l1_widths.pt"
M_GLOBAL_PATH = "./data/train/clusters/m_global.pt"

# 专属的保存文件夹，绝不干扰联合训练的权重！
SAVE_DIR = "./checkpoints/phase0_router_only"

def main():
    print("="*50)
    print("⚡ 启动 IH-PromptDSI [第二阶段：Router 离线暴刷引擎]")
    print("="*50)

    # 1. 加载精华特征
    print(f">> 正在从 {FEATURES_PATH} 加载 11 万条精华数据...")
    dataset_dict = torch.load(FEATURES_PATH, map_location="cpu", weights_only=False)
    
    q_vecs = dataset_dict["q_vecs"].to(torch.bfloat16)
    l2_targets = dataset_dict["l2_targets"].long()
    
    dataset = TensorDataset(q_vecs, l2_targets)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    print(f"✅ 数据加载成功！共 {len(dataset)} 条，Batch Size = {BATCH_SIZE}")

    # 2. 重建 Router 和 MLP 脑区
    print(">> 正在组装 Router 和 特征降维层...")
    l2_centers = torch.load(L2_CENTERS_PATH, map_location=DEVICE, weights_only=False).to(torch.bfloat16)
    l2_widths = torch.load(L2_WIDTHS_PATH, map_location=DEVICE, weights_only=False).to(torch.bfloat16)
    l1_centers = torch.load(L1_CENTERS_PATH, map_location=DEVICE, weights_only=False).to(torch.bfloat16)
    l1_widths = torch.load(L1_WIDTHS_PATH, map_location=DEVICE, weights_only=False).to(torch.bfloat16)
    m_global = torch.load(M_GLOBAL_PATH, map_location=DEVICE, weights_only=False).to(torch.bfloat16)
    
    num_boxes = l2_centers.shape[0] 
    dim = l2_centers.shape[1] 

    router = DualStreamRouter(
        dim=dim, num_tools=NUM_TOOLS, num_boxes=num_boxes, m_global=m_global,
        l2_centers=l2_centers, l2_widths=l2_widths, l1_centers=l1_centers, l1_widths=l1_widths
    ).to(DEVICE)

    query_proj = nn.Sequential(
        nn.Linear(4096, 1024, dtype=torch.bfloat16),
        nn.GELU(),
        nn.Dropout(0.15),
        nn.Linear(1024, 384, dtype=torch.bfloat16)
    ).to(DEVICE)

    # 3. 优化器和损失函数
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = AdamW(list(router.parameters()) + list(query_proj.parameters()), lr=2e-3) # 猛烈轰炸

    # 4. 开启暴力狂飙模式
    print("🔥 引擎点火，开始暴力拟合！\n" + "-"*50)
    router.train()
    query_proj.train()

    for epoch in range(EPOCHS):
        total_loss, correct_top1, correct_top5, total_samples = 0.0, 0, 0, 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", ascii=True)
        for batch_q, batch_target in pbar:
            optimizer.zero_grad()
            
            batch_q = batch_q.to(DEVICE)
            batch_target = batch_target.to(DEVICE)
            t_prev = torch.full_like(batch_target, -1).to(DEVICE)

            q_projected = query_proj(batch_q)
            s_total, _ = router(q_projected, t_prev)
            s_total = s_total * 15.0
            loss = loss_fn(s_total, batch_target)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            with torch.no_grad():
                top5_preds = s_total.topk(5, dim=-1)[1]
                correct_top1 += (top5_preds[:, 0] == batch_target).sum().item()
                correct_top5 += (top5_preds == batch_target.unsqueeze(1)).sum().item()
                total_samples += batch_target.size(0)

            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Top1%": f"{(correct_top1/total_samples)*100:.1f}",
                "Top5%": f"{(correct_top5/total_samples)*100:.1f}"
            }, refresh=False)

        print(f"=== Epoch {epoch} 总结 ===")
        print(f"🎯 Router Loss: {total_loss/len(dataloader):.4f} | Top-1 命中: {(correct_top1/total_samples)*100:.2f}% | Top-5 命中: {(correct_top5/total_samples)*100:.2f}%")

    # 5. 保存到全新规划的独立目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "offline_router_master.pt")
    torch.save({
        'router_state_dict': router.state_dict(),
        'query_proj_state_dict': query_proj.state_dict(),
    }, save_path)
    
    print("="*50)
    print(f"🏆 离线速成完毕！神级 Router 权重已保存至: {save_path}")

if __name__ == "__main__":
    main()