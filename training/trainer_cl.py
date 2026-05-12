import torch
import os
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
import random
import datetime

_DEBUG = int(os.environ.get("DEBUG_NAN", "0"))


class UnifiedSystem(nn.Module):
    def __init__(self, router, llm_caller):
        super().__init__()
        self.router = router
        self.llm_caller = llm_caller

    def forward(self, query_texts, t_prev, train_l2_id, target_l1, target_l2, _debug=False, _semi_freeze_off=False):
        q_vec, input_ids, attention_mask = self.llm_caller.extract_query_vector(query_texts)
        s_total, s_sem = self.router(q_vec, t_prev)
        logits = self.llm_caller(input_ids, attention_mask, train_l2_id)

        # --- 数值安全 clamp：semi_freeze_off 时，所有输出 clamp 到安全范围 ---
        if _semi_freeze_off:
            logits = torch.clamp(logits, min=-30.0, max=30.0)
            s_total = torch.clamp(s_total, min=1e-7, max=1.0)

        # --- NaN 定位 debug ---
        if _debug:
            for _n, _t in [("q_vec", q_vec), ("s_total", s_total), ("s_sem", s_sem), ("logits", logits)]:
                if torch.isnan(_t).any():
                    print(f"\n  [NaN] {_n}: {torch.isnan(_t).sum()}/{_t.numel()} NaN, shape={_t.shape}, dtype={_t.dtype}")
                    print(f"    min={_t.nan_to_num(nan=0).amin():.4f} max={_t.amin():.4f}")

        # LinearBoxRouter 和 FlatDualStreamRouter 都没有 l1_centers/l1_widths，跳过几何损失输入
        if hasattr(self.router, 'l1_centers'):
            l1_c = self.router.l1_centers[target_l1]
            l1_w = self.router.l1_widths[target_l1]
        else:
            l1_c = None
            l1_w = None

        # 只有 DualStreamRouter 有 l2_centers/l2_widths（LinearBoxRouter 用 Linear 替代）
        if hasattr(self.router, 'l2_centers'):
            l2_c = self.router.l2_centers[target_l2]
            l2_w = self.router.l2_widths[target_l2]
        else:
            l2_c = None
            l2_w = None

        return s_total, logits, l1_c, l1_w, l2_c, l2_w


class IH_Trainer:
    def __init__(self, router, llm_caller, loss_fn, dataloader, device, accelerator, semi_freeze_off=False):
        self.device = device
        self.accelerator = accelerator
        self.router = router
        self.llm_caller = llm_caller
        self._semi_freeze_off = semi_freeze_off

        # 判断 router 是否有几何信息（用于几何损失计算）
        if hasattr(router, 'l2_centers'):
            self._router_has_geo = True
        elif hasattr(router, 'prompt_proj'):
            self._router_has_geo = False
        else:
            self._router_has_geo = False

        # 🚨 动态收集：自动收集所有 requires_grad=True 的参数
        # 这样 semi_freeze_off 时解冻的参数会自动被优化器训练
        trainable_params = (
            [p for p in router.parameters() if p.requires_grad] +
            [p for p in llm_caller.parameters() if p.requires_grad]
        )
        
        # 🚨 分组学习率：semi_freeze_off 时需要特别处理
        # semi_freeze_off 全参数微调时，几何参数 l2_centers 需要更小的 lr 防止被冲走
        # 正常训练时只有 l2_centers 可训练，沿用原来的 lr=1e-3
        param_groups = []
        
        if self._semi_freeze_off:
            # semi_freeze_off：几何参数用小 lr，其他参数用正常 lr
            if hasattr(router, 'l2_centers') and router.l2_centers.requires_grad:
                param_groups.append({
                    "params": [router.l2_centers],
                    "lr": 1e-5,
                    "weight_decay": 0.0  # 几何参数不加 weight_decay
                })
            
            other_params = [p for p in trainable_params 
                          if not (hasattr(router, 'l2_centers') and p is router.l2_centers)]
            if other_params:
                param_groups.append({
                    "params": other_params,
                    "lr": 1e-3,
                    "weight_decay": 0.01
                })
        else:
            # 正常训练：所有参数共用 lr（和原来一样）
            param_groups = [{"params": trainable_params, "lr": 1e-3, "weight_decay": 0.01}]
        
        self.optimizer = AdamW(param_groups)

        self.unified_model = UnifiedSystem(router, llm_caller)
        self.unified_model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.unified_model, self.optimizer, dataloader
        )

        self.loss_fn = loss_fn
        self.lambda_1 = 0.5
        self.lambda_2 = 5.0

        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.last_best_ckpt_path = None

        self.num_boxes = router.num_boxes
        self.ce_loss_fn = nn.CrossEntropyLoss().to(self.device)
        # LinearBoxRouter 没有 l2_centers，跳过半冻结备份/恢复
        if hasattr(router, 'l2_centers'):
            self.old_boxes_backup = router.l2_centers.data[:llm_caller.old_num_boxes].clone().detach()
        else:
            self.old_boxes_backup = None

        self.patience = 3
        self.best_loss = float('inf')
        self.patience_counter = 0

        # 学习率热启动：semi_freeze_off 时，前 2 epoch 用更小的 lr 防止不稳定
        self._orig_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        self._lr_warmup_done = False

        # 扁平空间没有 L1 层，几何损失恒为 0
        self._has_l1 = hasattr(router, 'l1_centers')

    def train_epoch(self, epoch_idx):
        self.unified_model.train()

        # 学习率热启动：前 2 epoch 用极小的 lr，第 3 epoch 恢复正常
        if self._semi_freeze_off:
            if epoch_idx < 2:
                for pg in self.optimizer.param_groups:
                    # 几何参数预热 lr=1e-6，其他参数预热 lr=1e-5
                    if pg.get("weight_decay", 0.01) == 0.0:  # 几何参数
                        pg["lr"] = 1e-6
                    else:
                        pg["lr"] = 1e-5
            elif not self._lr_warmup_done:
                for pg, orig_lr in zip(self.optimizer.param_groups, self._orig_lrs):
                    pg["lr"] = orig_lr
                self._lr_warmup_done = True

        total_loss, task_loss_sum, geo_loss_sum, cont_loss_sum = 0.0, 0.0, 0.0, 0.0
        valid_batches = 0
        if self.accelerator.is_main_process:
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch_idx}", leave=True, dynamic_ncols=True)
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
                s_total, logits, l1_c, l1_w, l2_c, l2_w = self.unified_model(
                    query_texts, t_prev, train_l2_id, target_l1, target_l2,
                    _debug=(_DEBUG and batch_idx < 3),
                    _semi_freeze_off=self._semi_freeze_off
                )

                l_task_base = self.ce_loss_fn(logits, tool_label)

                # 扁平空间无 L1 层，几何损失为 0
                if self._has_l1:
                    l_geo = self.loss_fn.compute_geo_loss(l1_c, l1_w, l2_c, l2_w).to(self.device)
                else:
                    l_geo = torch.tensor(0.0, device=self.device)

                l_contrastive = self.loss_fn.compute_contrastive_loss(s_total, target_l2).to(self.device)

                l_task = l_task_base

                # NaN 安全保护：跳过含 NaN 的 batch，防止权重被污染
                skip_batch = (
                    torch.isnan(l_task).any() if isinstance(l_task, torch.Tensor) else torch.isnan(torch.tensor(l_task)).any() or
                    torch.isnan(l_geo).any() or
                    torch.isnan(l_contrastive).any()
                )
                if skip_batch:
                    if self.accelerator.is_main_process and batch_idx < 5:
                        print(f"\n⚠️  [b{batch_idx}] Loss=NaN (task={float(l_task):.4f}, geo={float(l_geo):.4f}, cont={float(l_contrastive):.4f})，跳过该 batch")
                    self.optimizer.zero_grad()
                    continue

                loss = l_task + self.lambda_1 * l_geo + self.lambda_2 * l_contrastive

                self.accelerator.backward(loss)

                if self._router_has_geo and self.router.l2_centers.grad is not None:
                    old_boxes = getattr(self.llm_caller, 'old_num_boxes', 0)
                    if old_boxes > 0:
                        self.router.l2_centers.grad[:old_boxes] = 0.0

                # 梯度裁剪：对所有 param_groups 分别裁剪
                for pg in self.optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(pg['params'], max_norm=0.5)
                self.optimizer.step()

                with torch.no_grad():
                    if self._router_has_geo and self.old_boxes_backup is not None:
                        self.router.l2_centers.data[:len(self.old_boxes_backup)] = self.old_boxes_backup

            total_loss += loss.item()
            task_loss_sum += l_task.item()
            geo_loss_sum += l_geo.item()
            cont_loss_sum += l_contrastive.item()
            valid_batches += 1

            if self.accelerator.is_main_process:
                pbar.set_postfix({"Loss": f"{loss.item():.3f}"})

        if self.accelerator.is_main_process:
            if valid_batches == 0:
                print(f"=== Epoch {epoch_idx} 完成 ===")
                print(f"⚠️  全部 {len(pbar)} batch 均为 NaN，无法计算平均 loss，跳过 checkpoint 保存")
                return 0.0, 0.0, 0.0, 0.0, False, False

            num_batches = valid_batches
            avg_loss = total_loss / num_batches
            print(f"=== Epoch {epoch_idx} 完成 ===")
            print(f"平均总 Loss: {avg_loss:.4f}（有效 batch: {valid_batches}/{len(pbar)}）")

            early_stop = False
            is_best = False

            if avg_loss < self.best_loss and avg_loss > 0:
                self.best_loss = avg_loss
                self.patience_counter = 0
                is_best = True
            else:
                self.patience_counter += 1
                print(f"⚠️ Loss 遇平台期，早停计数: {self.patience_counter}/{self.patience}")
                if self.patience_counter >= self.patience:
                    print("🛑 触发早停机制！(Early Stopping) 即将终止训练！")
                    early_stop = True

            return avg_loss, task_loss_sum / num_batches, geo_loss_sum / num_batches, cont_loss_sum / num_batches, early_stop, is_best

        return 0, 0, 0, 0, False, False

    def save_checkpoint(self, epoch_idx, save_dir="checkpoints/task1"):
        os.makedirs(save_dir, exist_ok=True)
        unwrapped_system = self.accelerator.unwrap_model(self.unified_model)
        unwrapped_router = unwrapped_system.router
        unwrapped_llm_caller = unwrapped_system.llm_caller

        if hasattr(unwrapped_llm_caller, 'prompt_pool_new'):
            full_prompt_pool = torch.cat([
                unwrapped_llm_caller.prompt_pool_old.data,
                unwrapped_llm_caller.prompt_pool_new.data
            ], dim=0)
        else:
            full_prompt_pool = unwrapped_llm_caller.prompt_pool.data

        # 兼容 CL 版本（classifier_old/new）和基线版本（单一 classifier）
        if hasattr(unwrapped_llm_caller, 'classifier_old'):
            classifier_state_dict = {
                'weight_old': unwrapped_llm_caller.classifier_old.weight.data,
                'bias_old':   unwrapped_llm_caller.classifier_old.bias.data,
                'weight_new': unwrapped_llm_caller.classifier_new.weight.data,
                'bias_new':   unwrapped_llm_caller.classifier_new.bias.data,
            }
        else:
            classifier_state_dict = {
                'weight': unwrapped_llm_caller.classifier.weight.data,
                'bias':   unwrapped_llm_caller.classifier.bias.data
            }

        checkpoint = {
            'epoch': epoch_idx,
            'router_state_dict': unwrapped_router.state_dict(),
            'query_proj_state_dict': unwrapped_llm_caller.query_proj.state_dict(),
            'prompt_pool': full_prompt_pool.cpu(),
            'classifier_state_dict': {k: v.cpu() for k, v in classifier_state_dict.items()},
            'optimizer_state_dict': self.optimizer.state_dict(),
            'old_num_boxes': unwrapped_llm_caller.old_num_boxes,
            'old_num_tools': unwrapped_llm_caller.old_num_tools,
            # 早停相关状态（用于断点续训）
            'best_loss': self.best_loss,
            'patience_counter': self.patience_counter,
        }

        last_path = getattr(self, "last_best_ckpt_path", None)
        if last_path and os.path.exists(last_path):
            try:
                os.remove(last_path)
            except Exception:
                pass

        ts = getattr(self, "timestamp", "latest")
        filename = f"ih_prompt_dsi_cls_best_epoch_{epoch_idx}_{ts}.pt"
        path = os.path.join(save_dir, filename)

        torch.save(checkpoint, path)
        self.last_best_ckpt_path = path
        print(f"\n💾 破纪录！已保存最佳权重: {filename}")

    def load_checkpoint(self, checkpoint_path, optimizer_path=None):
        """
        从 checkpoint 加载权重和优化器状态。

        Args:
            checkpoint_path: checkpoint .pt 文件路径
            optimizer_path:  如果传入 optimizer 状态独立保存的文件路径，则同时加载

        Returns:
            epoch_idx: 从 checkpoint 中恢复的 epoch 编号
        """
        print(f">> 加载 checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        unwrapped_system = self.accelerator.unwrap_model(self.unified_model)
        unwrapped_router = unwrapped_system.router
        unwrapped_llm_caller = unwrapped_system.llm_caller

        # 加载 router 权重
        unwrapped_router.load_state_dict(ckpt['router_state_dict'])

        # 加载 query_proj 权重
        unwrapped_llm_caller.query_proj.load_state_dict(ckpt['query_proj_state_dict'])

        # 加载 optimizer 状态（如果有）
        if 'optimizer_state_dict' in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                print(f">> 优化器状态已恢复")
            except Exception as e:
                print(f">> ⚠️  优化器状态加载失败（可能是参数不匹配），将使用当前初始化: {e}")

        # 恢复早停相关状态
        if 'best_loss' in ckpt:
            self.best_loss = ckpt['best_loss']
        if 'patience_counter' in ckpt:
            self.patience_counter = ckpt['patience_counter']

        epoch_idx = ckpt.get('epoch', 0)
        print(f">> 权重已恢复（epoch={epoch_idx}, best_loss={self.best_loss:.4f}）")

        return epoch_idx
