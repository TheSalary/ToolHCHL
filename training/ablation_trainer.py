"""
training/ablation_trainer.py
============================
消融实验统一训练器。

从 run_ablation.py 的 run_train_base / run_train_cl 提取并重构，
依赖 models/router.py 和 models/llm_caller.py 的新工厂体系。

主要功能
--------
- 统一的 BaseTrainer / CLTrainer，支持所有消融实验模式
- 与 models.router.create_router / models.llm_caller.create_llm_caller 无缝集成
- 统一的 save_checkpoint / load_checkpoint 逻辑

使用示例
--------
    from training.ablation_trainer import BaseTrainer, CLTrainer

    trainer = CLTrainer(
        router=router, llm_caller=llm_caller,
        loss_fn=loss_fn, dataloader=dataloader,
        device=DEVICE, accelerator=accelerator,
        ablation_cfg={"geo_off": False, "semi_freeze_off": False}
    )
"""

import os
import random
import datetime
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from training.ablation_components import AblationLoss
from training.losses import IHLoss


# ============================================================================
# 1. UnifiedSystem — 统一前向系统
# ============================================================================
class UnifiedSystem(nn.Module):
    """
    Router + LLMCaller 统一前向系统。

    封装 Router 和 LLMCaller 的联合前向逻辑，兼容所有 Router 类型：
    - DualStreamRouter（有 l1_centers，支持几何损失）
    - FlatDualStreamRouter（有 l2_centers，无 L1 层）
    - SimpleLinearRouter（无几何参数，几何损失恒为 0）
    """

    def __init__(self, router, llm_caller):
        super().__init__()
        self.router = router
        self.llm_caller = llm_caller

    def forward(self, query_texts, t_prev, train_l2_id, target_l1, target_l2,
                _debug=False, _semi_freeze_off=False):
        q_vec, input_ids, attention_mask = self.llm_caller.extract_query_vector(query_texts)
        s_total, s_sem = self.router(q_vec, t_prev)
        logits = self.llm_caller(input_ids, attention_mask, train_l2_id)

        # 数值安全 clamp：semi_freeze_off 全参数微调时防止数值爆炸
        if _semi_freeze_off:
            logits = torch.clamp(logits, min=-30.0, max=30.0)
            s_total = torch.clamp(s_total, min=1e-7, max=1.0)

        if _debug:
            for _n, _t in [("q_vec", q_vec), ("s_total", s_total),
                            ("s_sem", s_sem), ("logits", logits)]:
                if torch.isnan(_t).any():
                    print(f"\n  [NaN] {_n}: {torch.isnan(_t).sum()}/{_t.numel()}")

        # 几何损失输入（根据 Router 类型决定是否提供）
        has_l1 = hasattr(self.router, 'l1_centers')
        has_l2 = hasattr(self.router, 'l2_centers')

        if has_l1:
            l1_c = self.router.l1_centers[target_l1]
            l1_w = self.router.l1_widths[target_l1]
        else:
            l1_c = None
            l1_w = None

        if has_l2:
            l2_c = self.router.l2_centers[target_l2]
            l2_w = self.router.l2_widths[target_l2]
        else:
            l2_c = None
            l2_w = None

        return s_total, logits, l1_c, l1_w, l2_c, l2_w


# ============================================================================
# 2. BaseTrainer — 基线训练器
# ============================================================================
class BaseTrainer:
    """
    基线版训练器（对应 Base 任务）。

    适用于单一 prompt_pool + 单一 classifier 的 LLMCallerBase。
    支持的消融实验：geo_loss_off、linear_router
    """

    def __init__(self, router, llm_caller, loss_fn, dataloader,
                 device, accelerator, ablation_cfg=None):
        self.device = device
        self.accelerator = accelerator
        self.router = router
        self.llm_caller = llm_caller
        self.loss_fn = loss_fn
        self.ablation_cfg = ablation_cfg or {}

        # 收集可训练参数
        trainable_params = list(llm_caller.classifier.parameters()) + [llm_caller.prompt_pool]
        if hasattr(router, 'prompt_proj'):
            trainable_params.extend(list(router.prompt_proj.parameters()))
        self._has_router_trainable = hasattr(router, 'prompt_proj')

        self.optimizer = AdamW(trainable_params, lr=1e-4, weight_decay=0.01)

        self.unified_model = UnifiedSystem(router, llm_caller)
        self.unified_model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.unified_model, self.optimizer, dataloader,
        )

        self.lambda_1 = 0.5
        self.lambda_2 = 5.0
        self.num_boxes = router.num_boxes
        self.ce_loss_fn = nn.CrossEntropyLoss().to(self.device)

    def train_epoch(self, epoch_idx):
        self.unified_model.train()
        total_loss, task_loss_sum, geo_loss_sum, cont_loss_sum = 0.0, 0.0, 0.0, 0.0

        if self.accelerator.is_main_process:
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch_idx}",
                        leave=True, dynamic_ncols=True, ascii=True)
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
                )

                l_task = self.ce_loss_fn(logits, tool_label)

                has_geo = hasattr(self.router, 'l1_centers')
                l_geo = self.loss_fn.compute_geo_loss(
                    l1_c, l1_w, l2_c, l2_w
                ).to(self.device) if has_geo else torch.tensor(0.0, device=self.device)

                l_contrastive = self.loss_fn.compute_contrastive_loss(
                    s_total, target_l2
                ).to(self.device)

                loss = l_task + self.lambda_1 * l_geo + self.lambda_2 * l_contrastive
                self.accelerator.backward(loss)

                torch.nn.utils.clip_grad_norm_(
                    self.optimizer.param_groups[0]['params'], max_norm=1.0,
                )
                self.optimizer.step()

            total_loss += loss.item()
            task_loss_sum += l_task.item()
            geo_loss_sum += l_geo.item()
            cont_loss_sum += l_contrastive.item()

            if self.accelerator.is_main_process:
                pbar.set_postfix({
                    "Loss": f"{loss.item():.3f}",
                    "Task": f"{l_task.item():.3f}",
                    "Cont": f"{l_contrastive.item():.3f}",
                }, refresh=False)

        if self.accelerator.is_main_process:
            num_batches = len(self.train_dataloader)
            print(f"=== Epoch {epoch_idx} 完成 ===")
            print(f"平均总 Loss: {total_loss / num_batches:.4f}")
            print(f"-> 分类: {task_loss_sum / num_batches:.4f} | "
                  f"几何: {geo_loss_sum / num_batches:.4f} | "
                  f"对比: {cont_loss_sum / num_batches:.4f}")
            return (total_loss / num_batches, task_loss_sum / num_batches,
                    geo_loss_sum / num_batches, cont_loss_sum / num_batches)
        return 0, 0, 0, 0

    def save_checkpoint(self, epoch_idx, save_dir="checkpoints", suffix=""):
        os.makedirs(save_dir, exist_ok=True)
        unwrapped = self.accelerator.unwrap_model(self.unified_model)
        router = unwrapped.router
        llm = unwrapped.llm_caller

        checkpoint = {
            'epoch': epoch_idx,
            'router_state_dict': router.state_dict(),
            'query_proj_state_dict': llm.query_proj.state_dict(),
            'prompt_pool': llm.prompt_pool.detach().cpu(),
            'classifier_state_dict': llm.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        name = f"ih_prompt_dsi_cls_epoch_{epoch_idx}{suffix}.pt"
        torch.save(checkpoint, os.path.join(save_dir, name))

    def load_checkpoint(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        unwrapped = self.accelerator.unwrap_model(self.unified_model)
        router = unwrapped.router
        llm = unwrapped.llm_caller

        router.load_state_dict(ckpt['router_state_dict'])
        llm.query_proj.load_state_dict(ckpt['query_proj_state_dict'])
        llm.classifier.load_state_dict(ckpt['classifier_state_dict'])
        llm.prompt_pool.data = ckpt['prompt_pool'].to(self.device)

        if 'optimizer_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        epoch = ckpt.get('epoch', -1)
        print(f"✅ 已从 epoch={epoch} 的 checkpoint 恢复: {checkpoint_path}")
        return epoch


# ============================================================================
# 3. CLTrainer — 持续学习训练器
# ============================================================================
class CLTrainer:
    """
    持续学习版训练器（对应 Task1/2/3）。

    适用于 old/new 分离的 LLMCallerCL。
    支持的消融实验：
        - geo_loss_off：关闭几何包含损失
        - semi_freeze_off：关闭物理半冻结（全参数微调）
        - weight_inherit_off：跳过旧 checkpoint（随机初始化新参数）
        - replay_off / replay_per_tool：关闭或调整经验回放比例
        - linear_router / w/o_hierarchy：使用扁平/线性路由器

    训练策略
    --------
    - 默认：物理半冻结（仅 l2_centers 新盒子部分可训练）
    - semi_freeze_off：全参数可训练，含 l2_centers 几何学习率分离
    - 动态收集所有 requires_grad=True 的参数（自动适应各种半冻结策略）
    - NaN 安全保护：跳过含 NaN 的 batch
    - 早停机制：连续 N 个 epoch 无改善则停止
    """

    def __init__(self, router, llm_caller, loss_fn, dataloader,
                 device, accelerator, ablation_cfg=None):
        self.device = device
        self.accelerator = accelerator
        self.router = router
        self.llm_caller = llm_caller
        self.loss_fn = loss_fn
        self.ablation_cfg = ablation_cfg or {}
        self._semi_freeze_off = self.ablation_cfg.get("semi_freeze_off", False)

        # 动态收集所有 requires_grad=True 的参数
        trainable_params = (
            [p for p in router.parameters() if p.requires_grad] +
            [p for p in llm_caller.parameters() if p.requires_grad]
        )

        # 分组学习率：semi_freeze_off 时几何参数用更小的 lr
        param_groups = []
        if self._semi_freeze_off:
            if hasattr(router, 'l2_centers') and router.l2_centers.requires_grad:
                param_groups.append({
                    "params": [router.l2_centers],
                    "lr": 1e-5,
                    "weight_decay": 0.0,
                })
            other = [p for p in trainable_params
                     if not (hasattr(router, 'l2_centers') and p is router.l2_centers)]
            if other:
                param_groups.append({"params": other, "lr": 1e-3, "weight_decay": 0.01})
        else:
            param_groups = [{"params": trainable_params, "lr": 1e-3, "weight_decay": 0.01}]

        self.optimizer = AdamW(param_groups)

        self.unified_model = UnifiedSystem(router, llm_caller)
        self.unified_model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.unified_model, self.optimizer, dataloader,
        )

        self.lambda_1 = 0.5
        self.lambda_2 = 5.0
        self.num_boxes = router.num_boxes
        self.ce_loss_fn = nn.CrossEntropyLoss().to(self.device)

        # 半冻结备份
        if hasattr(router, 'l2_centers'):
            n_old = getattr(llm_caller, 'old_num_boxes', 0)
            self.old_boxes_backup = router.l2_centers.data[:n_old].clone().detach()
        else:
            self.old_boxes_backup = None

        # 早停
        self.patience = 3
        self.best_loss = float('inf')
        self.patience_counter = 0

        # 热启动 lr
        self._orig_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        self._lr_warmup_done = False

        # L1 层是否存在（决定是否计算几何损失）
        self._has_l1 = hasattr(router, 'l1_centers')

        # 路由是否有几何信息
        self._router_has_geo = (
            hasattr(router, 'l2_centers') or hasattr(router, 'prompt_proj')
        )

        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.last_best_ckpt_path = None

    def train_epoch(self, epoch_idx):
        self.unified_model.train()

        # 热启动 lr（semi_freeze_off 前 2 epoch）
        if self._semi_freeze_off:
            if epoch_idx < 2:
                for pg in self.optimizer.param_groups:
                    if pg.get("weight_decay", 0.01) == 0.0:
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
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch_idx}",
                        leave=True, dynamic_ncols=True)
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
                    _semi_freeze_off=self._semi_freeze_off,
                )

                l_task_base = self.ce_loss_fn(logits, tool_label)

                if self._has_l1 and l1_c is not None:
                    l_geo = self.loss_fn.compute_geo_loss(l1_c, l1_w, l2_c, l2_w).to(self.device)
                else:
                    l_geo = torch.tensor(0.0, device=self.device)

                l_contrastive = self.loss_fn.compute_contrastive_loss(
                    s_total, target_l2,
                ).to(self.device)

                l_task = l_task_base

                # NaN 安全保护
                skip = (
                    torch.isnan(l_task).any() if isinstance(l_task, torch.Tensor)
                    else torch.isnan(torch.tensor(l_task)).any()
                    or torch.isnan(l_geo).any()
                    or torch.isnan(l_contrastive).any()
                )
                if skip:
                    if self.accelerator.is_main_process and batch_idx < 5:
                        print(f"\n⚠️  [b{batch_idx}] Loss=NaN，跳过该 batch")
                    self.optimizer.zero_grad()
                    continue

                loss = l_task + self.lambda_1 * l_geo + self.lambda_2 * l_contrastive
                self.accelerator.backward(loss)

                # 梯度清零：老盒子的 l2_centers 不更新
                if self._router_has_geo and self.router.l2_centers.grad is not None:
                    n_old = getattr(self.llm_caller, 'old_num_boxes', 0)
                    if n_old > 0:
                        self.router.l2_centers.grad[:n_old] = 0.0

                for pg in self.optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(pg['params'], max_norm=0.5)
                self.optimizer.step()

                # 半冻结恢复：每个 step 结束后将老盒子恢复到备份值
                if self._router_has_geo and self.old_boxes_backup is not None:
                    self.router.l2_centers.data[:len(self.old_boxes_backup)] = \
                        self.old_boxes_backup

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
                print(f"⚠️  全部 batch 均为 NaN，跳过 checkpoint 保存")
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
                print(f"⚠️ 早停计数: {self.patience_counter}/{self.patience}")
                if self.patience_counter >= self.patience:
                    print("🛑 触发早停机制！")
                    early_stop = True

            return (avg_loss, task_loss_sum / num_batches,
                    geo_loss_sum / num_batches, cont_loss_sum / num_batches,
                    early_stop, is_best)

        return 0, 0, 0, 0, False, False

    def save_checkpoint(self, epoch_idx, save_dir="checkpoints"):
        os.makedirs(save_dir, exist_ok=True)
        unwrapped = self.accelerator.unwrap_model(self.unified_model)
        router = unwrapped.router
        llm = unwrapped.llm_caller

        # prompt_pool 拼接
        if hasattr(llm, 'prompt_pool_new'):
            full_prompt_pool = torch.cat([
                llm.prompt_pool_old.data,
                llm.prompt_pool_new.data,
            ], dim=0)
        else:
            full_prompt_pool = llm.prompt_pool.data

        # classifier 状态
        if hasattr(llm, 'classifier_old'):
            classifier_state_dict = {
                'weight_old': llm.classifier_old.weight.data,
                'bias_old':   llm.classifier_old.bias.data,
                'weight_new': llm.classifier_new.weight.data,
                'bias_new':   llm.classifier_new.bias.data,
            }
        else:
            classifier_state_dict = {
                'weight': llm.classifier.weight.data,
                'bias':   llm.classifier.bias.data,
            }

        checkpoint = {
            'epoch': epoch_idx,
            'router_state_dict': router.state_dict(),
            'query_proj_state_dict': llm.query_proj.state_dict(),
            'prompt_pool': full_prompt_pool.cpu(),
            'classifier_state_dict': {k: v.cpu() for k, v in classifier_state_dict.items()},
            'optimizer_state_dict': self.optimizer.state_dict(),
            'old_num_boxes': llm.old_num_boxes,
            'old_num_tools': llm.old_num_tools,
            'best_loss': self.best_loss,
            'patience_counter': self.patience_counter,
        }

        # 删除旧最佳 checkpoint
        last = getattr(self, "last_best_ckpt_path", None)
        if last and os.path.exists(last):
            try:
                os.remove(last)
            except Exception:
                pass

        filename = f"ih_prompt_dsi_cls_best_epoch_{epoch_idx}_{self.timestamp}.pt"
        path = os.path.join(save_dir, filename)
        torch.save(checkpoint, path)
        self.last_best_ckpt_path = path
        print(f"\n💾 最佳权重已保存: {filename}")

    def load_checkpoint(self, checkpoint_path):
        print(f">> 加载 checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        unwrapped = self.accelerator.unwrap_model(self.unified_model)
        router = unwrapped.router
        llm = unwrapped.llm_caller

        router.load_state_dict(ckpt['router_state_dict'])
        llm.query_proj.load_state_dict(ckpt['query_proj_state_dict'])

        if 'optimizer_state_dict' in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                print(f">> 优化器状态已恢复")
            except Exception as e:
                print(f">> ⚠️ 优化器状态加载失败: {e}")

        if 'best_loss' in ckpt:
            self.best_loss = ckpt['best_loss']
        if 'patience_counter' in ckpt:
            self.patience_counter = ckpt['patience_counter']

        epoch = ckpt.get('epoch', 0)
        print(f">> 权重已恢复（epoch={epoch}, best_loss={self.best_loss:.4f}）")
        return epoch


# ============================================================================
# 4. 工厂函数
# ============================================================================
def create_ablation_loss(
    geo_off: bool = False,
    contrast_off: bool = False,
) -> AblationLoss:
    """
    创建消融版损失函数。

    参数:
        geo_off:      关闭几何包含损失
        contrast_off: 关闭路由对比损失
    """
    return AblationLoss(IHLoss(), geo_off=geo_off, contrast_off=contrast_off)


def build_ablation_cfg(args) -> dict:
    """
    从 argparse.Namespace 构建消融配置字典。

    支持两种指定方式：
    1. --ablation <name>            （自动解析）
    2. --geo-loss-off / --semi-freeze-off 等独立 flag

    返回:
        dict，含 geo_off、contrast_off、semi_freeze_off 等布尔字段
    """
    cfg = {}
    ablation = getattr(args, 'ablation', None)

    # 从 ablation 名称自动推导
    if ablation:
        if ablation == "geo_loss_off":
            cfg["geo_off"] = True
        if ablation == "w/o_hierarchy":
            cfg["router_type"] = "flat"
        if ablation == "flat_space":
            cfg["router_type"] = "flat"
        if ablation == "linear_router":
            cfg["router_type"] = "simple_linear"

    # 独立 flag（优先级更高）
    if getattr(args, 'geo_loss_off', False):
        cfg["geo_off"] = True
    if getattr(args, 'contrast_loss_off', False):
        cfg["contrast_off"] = True
    if getattr(args, 'semi_freeze_off', False):
        cfg["semi_freeze_off"] = True
    if getattr(args, 'weight_inherit_off', False):
        cfg["weight_inherit_off"] = True
    if getattr(args, 'replay_off', False):
        cfg["replay_off"] = True
    if getattr(args, 'replay_per_tool', 1) != 1:
        cfg["replay_per_tool"] = getattr(args, 'replay_per_tool', 1)

    return cfg
