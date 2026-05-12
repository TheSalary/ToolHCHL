import torch
import os
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
import random
import datetime

class UnifiedSystem(nn.Module):
    def __init__(self, router, llm_caller):
        super().__init__()
        self.router = router
        self.llm_caller = llm_caller

    def forward(self, query_texts, t_prev, train_l2_id, target_l1, target_l2):
        # 1. 变量名改成 input_ids，精准对接
        q_vec, input_ids, attention_mask = self.llm_caller.extract_query_vector(query_texts)

        # 2. 路由 (拿到盒子打分)
        s_total, s_sem = self.router(q_vec, t_prev)

        # 3. 传入 input_ids，让 llm_caller 内部自己去拼接 Prompt
        logits = self.llm_caller(input_ids, attention_mask, train_l2_id)

        # ... 后面的 DDP loss 计算保持不变 ...
        # LinearBoxRouter 没有 l1/l2 中心点，返回零占位
        if hasattr(self.router, 'l1_centers'):
            l1_c = self.router.l1_centers[target_l1]
            l1_w = self.router.l1_widths[target_l1]
            l2_c = self.router.l2_centers[target_l2]
            l2_w = self.router.l2_widths[target_l2]
        else:
            l1_c = torch.zeros_like(target_l1, dtype=torch.bfloat16)
            l1_w = torch.zeros_like(target_l1, dtype=torch.bfloat16)
            l2_c = torch.zeros_like(target_l2, dtype=torch.bfloat16)
            l2_w = torch.zeros_like(target_l2, dtype=torch.bfloat16)

        return s_total, logits, l1_c, l1_w, l2_c, l2_w

class IH_Trainer:
    def __init__(self, router, llm_caller, loss_fn, dataloader, device, accelerator):
        self.device = device
        self.accelerator = accelerator
        self.router = router
        self.llm_caller = llm_caller
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 收集可训练参数
        trainable_params = list(llm_caller.classifier.parameters()) + [llm_caller.prompt_pool]

        # LinearBoxRouter: 添加 prompt_proj 到可训练参数
        if hasattr(router, 'prompt_proj'):
            trainable_params.extend(list(router.prompt_proj.parameters()))
            self._has_router_trainable = True
        else:
            self._has_router_trainable = False

        self.optimizer = AdamW(trainable_params, lr=1e-4, weight_decay=0.01)
        
        self.unified_model = UnifiedSystem(router, llm_caller)
        self.unified_model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.unified_model, self.optimizer, dataloader
        )
        
        self.loss_fn = loss_fn
        self.lambda_1 = 0.5 
        self.lambda_2 = 5.0 
        self.num_boxes = 582
        self.ce_loss_fn = nn.CrossEntropyLoss().to(self.device)

    def train_epoch(self, epoch_idx):
        self.unified_model.train()
        
        total_loss, task_loss_sum, geo_loss_sum, cont_loss_sum = 0.0, 0.0, 0.0, 0.0
        if self.accelerator.is_main_process:
            pbar = tqdm(
                self.train_dataloader, 
                desc=f"Epoch {epoch_idx}", 
                leave=True,          # 跑完 Epoch 后把进度条留在屏幕上
                dynamic_ncols=True,   # 自动适应你的终端屏幕宽度
                ascii=True         # 👈 让日志文件少一些乱码
            )
        else:
            pbar = self.train_dataloader

        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()

            query_texts = batch["query_text"]
            t_prev = torch.full_like(batch["t_prev_id"], -1).to(self.device)
            target_l2 = batch["target_l2_id"].to(self.device)
            target_l1 = batch["target_l1_id"].to(self.device)
            tool_label = batch["tool_label"].to(self.device) 

            if random.random() < 0.15:
                train_l2_id = torch.randint(0, self.num_boxes, target_l2.shape).to(self.device)
            else:
                train_l2_id = target_l2

            with self.accelerator.accumulate(self.unified_model):
                # 🚨 大一统前向传播，接管所有参数
                s_total, logits, l1_c, l1_w, l2_c, l2_w = self.unified_model(
                    query_texts, t_prev, train_l2_id, target_l1, target_l2
                )

                l_task = self.ce_loss_fn(logits, tool_label)
                has_geo = hasattr(self.router, 'l1_centers')
                l_geo = self.loss_fn.compute_geo_loss(l1_c, l1_w, l2_c, l2_w).to(self.device) if has_geo else torch.tensor(0.0, device=self.device)
                l_contrastive = self.loss_fn.compute_contrastive_loss(s_total, target_l2).to(self.device)

                loss = l_task + self.lambda_1 * l_geo + self.lambda_2 * l_contrastive
                self.accelerator.backward(loss)

                torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()
            task_loss_sum += l_task.item()
            geo_loss_sum += l_geo.item()
            cont_loss_sum += l_contrastive.item()

            if self.accelerator.is_main_process:
                pbar.set_postfix({
                    "Loss": f"{loss.item():.3f}",
                    "Task": f"{l_task.item():.3f}",
                    "Cont": f"{l_contrastive.item():.3f}"
                }, refresh=False)

        if self.accelerator.is_main_process:
            num_batches = len(self.train_dataloader)
            print(f"=== Epoch {epoch_idx} 完成 ===")
            print(f"平均总 Loss: {total_loss / num_batches:.4f}")
            print(f"-> 分类任务: {task_loss_sum / num_batches:.4f} | 几何包含: {geo_loss_sum / num_batches:.4f} | 路由对比: {cont_loss_sum / num_batches:.4f}")

            return total_loss / num_batches, task_loss_sum / num_batches, geo_loss_sum / num_batches, cont_loss_sum / num_batches
        return 0, 0, 0, 0

    def save_checkpoint(self, epoch_idx, save_dir="checkpoints", suffix=""):
        os.makedirs(save_dir, exist_ok=True)
        unwrapped_system = self.accelerator.unwrap_model(self.unified_model)
        unwrapped_router = unwrapped_system.router
        unwrapped_llm_caller = unwrapped_system.llm_caller
        
        checkpoint = {
            'epoch': epoch_idx,
            'router_state_dict': unwrapped_router.state_dict(),
            'query_proj_state_dict': unwrapped_llm_caller.query_proj.state_dict(),
            'prompt_pool': unwrapped_llm_caller.prompt_pool.detach().cpu(),
            'classifier_state_dict': unwrapped_llm_caller.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        ts = getattr(self, "timestamp", "latest")
        filename = f"ih_prompt_dsi_cls_epoch_{epoch_idx}_{ts}{suffix}.pt"
        path = os.path.join(save_dir, filename)
        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path, optimizer_path=None):
        """
        从 checkpoint 加载权重和优化器状态。

        Args:
            checkpoint_path: checkpoint .pt 文件路径
            optimizer_path:  如果传入 optimizer 状态独立保存的文件路径，则同时加载
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        unwrapped = self.accelerator.unwrap_model(self.unified_model)
        router = unwrapped.router
        llm = unwrapped.llm_caller

        router.load_state_dict(ckpt['router_state_dict'])
        llm.query_proj.load_state_dict(ckpt['query_proj_state_dict'])
        llm.classifier.load_state_dict(ckpt['classifier_state_dict'])
        llm.prompt_pool.data = ckpt['prompt_pool'].to(self.device)

        if optimizer_path:
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
        elif 'optimizer_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        loaded_epoch = ckpt.get('epoch', -1)
        print(f"✅ 已从 epoch={loaded_epoch} 的 checkpoint 恢复训练: {checkpoint_path}")
        return loaded_epoch
