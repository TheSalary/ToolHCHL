#!/usr/bin/env python3
"""
run_ablation.py
================
IH-PromptDSI 消融实验统一入口脚本（基于 run.py 重构）。

CLI 设计原则：
  --mode     : 工作流程步骤（prepare / train / eval / all），与 run.py 一致
  --ablation : 消融实验名称（None=基线，或以下任一名称）
  --train-only / --eval-only : 工作流控制 flag（不再混入 --mode）

消融实验分类：
  【需要重新训练】
    --ablation semi_freeze_off    : D2 关闭物理半冻结（全参数微调）
    --ablation weight_inherit_off : D3 关闭权重继承（随机初始化）
    --ablation replay_off         : D4 关闭经验回放
    --ablation geo_loss_off       : E1 关闭几何包含损失 L_geo
    --ablation w/o_hierarchy      : E2 去掉 L1 层，仅用 L2 软距离盒子
    --ablation flat_space         : E3 使用单层扁平空间（无 L1/L2 层级）
--ablation linear_router      : E4 使用 SimpleLinearRouter（无 box 机制，纯线性投影）

  【仅推理时评估（无需训练）】
    --ablation router_semantic    : R1 仅语义流，去除依赖流和门控
    --ablation router_dependency  : R2 仅依赖流，去除语义流
    --ablation router_no_gate     : R3 去除门控，固定 alpha=0.5 融合

  【综合评估】
    --ablation ablation_compare   : 在所有模型×自身测试集上跑推理时消融对比

使用示例：
  # 基线训练 + 评估（无需 --ablation）
  python run_ablation.py --task task1 --mode train --train-only
  python run_ablation.py --task task1 --mode eval

  # 消融实验：训练
  python run_ablation.py --task task1 --mode train --ablation semi_freeze_off --train-only
  python run_ablation.py --task task1 --mode eval --ablation semi_freeze_off

  # 消融实验：训练 + 评估（全流程）
  python run_ablation.py --task task1 --mode all --ablation semi_freeze_off

  # 推理时消融（无需训练，直接评估）
  python run_ablation.py --task task1 --mode eval --ablation router_semantic

  # 综合对比（推理时消融）
  python run_ablation.py --task task1 --mode eval --ablation ablation_compare
"""

import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import warnings
import math
import re
import json
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置所有随机种子，确保消融实验可复现
SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed()

# GPU 控制：支持命令行参数 --gpu 0,1,2,3 或环境变量 CUDA_VISIBLE_DEVICES
import argparse as _ap
_parser_gpu = _ap.ArgumentParser(add_help=False)
_parser_gpu.add_argument("--gpu", type=str, default=None)
_args_gpu, _ = _parser_gpu.parse_known_args()
if _args_gpu.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _args_gpu.gpu

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
warnings.filterwarnings("ignore")

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from config import LLAMA_PATH, TASK_CONFIGS, ABLATION_CONFIGS, INFERENCE_ABLATIONS
from config import normalize, load_data_mappings, get_checkpoint_dir

from models.router import (
    DualStreamRouter, FlatDualStreamRouter, SimpleLinearRouter,
    create_router, ABLATION_TO_ROUTER_TYPE,
)
from models.llm_caller import (
    LLMCallerBase, LLMCallerCL,
    create_llm_caller,
)
from training.ablation_components import AblationRouter, AblationLoss
from training.losses import IHLoss
from training.ablation_trainer import BaseTrainer as BaseTrainerNew, CLTrainer as CLTrainerNew
from training.ablation_trainer import build_ablation_cfg, create_ablation_loss


# ============================================================================
# 辅助函数（normalize / load_data_mappings / get_checkpoint_dir 来自 config.py）
# ============================================================================

def _get_latest_checkpoint(ckpt_dir):
    """获取目录下 epoch 最大的 checkpoint 文件名"""
    if not os.path.exists(ckpt_dir):
        return None
    files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if not files:
        return None
    def _epoch(fname):
        m = re.search(r'epoch_(\d+)', fname)
        return int(m.group(1)) if m else -1
    best = sorted([f for f in files if "_best" in f], key=_epoch, reverse=True)
    if best:
        return best[0]
    return sorted(files, key=_epoch, reverse=True)[0]


def _is_training_done(ckpt_dir, expected_epochs):
    """判断训练是否已完成（检查最后一个 epoch 是否有 checkpoint）"""
    if not os.path.exists(ckpt_dir):
        return False
    files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if not files:
        return False
    def _epoch(fname):
        m = re.search(r'epoch_(\d+)', fname)
        return int(m.group(1)) if m else -1
    max_ep = max(_epoch(f) for f in files)
    return max_ep >= expected_epochs - 1


# ============================================================================
# Mode: prepare
# ============================================================================
def run_prepare(task_name, stage, num_gpus):
    import subprocess
    cmd = [sys.executable, "scripts/prepare_task_data.py", "--task", task_name, "--stage", str(stage)]
    print(f"\n{'='*50}")
    print(f"[Prepare] 任务={task_name} stage={stage}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*50}")
    result = subprocess.run(cmd, cwd=os.getcwd())
    if result.returncode != 0:
        print(f"❌ prepare 失败！returncode={result.returncode}")
        sys.exit(result.returncode)
    print(f"✅ prepare 完成！")


# ============================================================================
# Mode: train (Base)
# ============================================================================
def run_train_base(task_cfg, args, resume=None):
    ablation = getattr(args, 'ablation', None)
    ckpt_dir = get_checkpoint_dir(task_cfg["name"], ablation)

    print(f"\n{'='*50}")
    print(f"[Train] {task_cfg['name']} — Base 训练")
    if ablation:
        print(f"[Train] 消融模式: {ablation}")
    print(f"[Train] 保存目录: {ckpt_dir}")
    print(f"{'='*50}")

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(gradient_accumulation_steps=2, kwargs_handlers=[ddp_kwargs])
    DEVICE = accelerator.device

    if accelerator.is_main_process:
        print(f">> 设备: {DEVICE}")

    l2_centers = torch.load(task_cfg["l2_centers"],  map_location=DEVICE, weights_only=False).to(torch.bfloat16)
    l2_widths  = torch.load(task_cfg["l2_widths"],   map_location=DEVICE, weights_only=False).to(torch.bfloat16)
    l1_centers = torch.load(task_cfg["l1_centers"],  map_location=DEVICE, weights_only=False).to(torch.bfloat16)
    l1_widths  = torch.load(task_cfg["l1_widths"],   map_location=DEVICE, weights_only=False).to(torch.bfloat16)
    m_global   = torch.load(task_cfg["m_global"],    map_location=DEVICE, weights_only=False).to(torch.bfloat16)

    num_boxes = l2_centers.shape[0]
    dim = l2_centers.shape[1]
    num_tools = task_cfg["old_num_tools"]

    # --- Router（使用工厂函数）---
    router_type = ABLATION_TO_ROUTER_TYPE.get(ablation, "dual_stream")
    if ablation == "linear_router":
        router = create_router(
            router_type="simple_linear",
            dim=dim, num_tools=num_tools, num_boxes=num_boxes,
        ).to(DEVICE)
        if accelerator.is_main_process:
            print(f">> [E4 线性路由器] 使用 SimpleLinearRouter（无 box 机制）")
    else:
        router = create_router(
            router_type=router_type,
            dim=dim, num_tools=num_tools, num_boxes=num_boxes,
            m_global=m_global,
            l2_centers=l2_centers, l2_widths=l2_widths,
            l1_centers=l1_centers, l1_widths=l1_widths,
        ).to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    base_llm = AutoModelForCausalLM.from_pretrained(LLAMA_PATH, torch_dtype=torch.bfloat16).to(DEVICE)
    base_llm.eval()
    for param in base_llm.parameters():
        param.requires_grad = False

    llm_caller = create_llm_caller(
        caller_type="base",
        base_llm=base_llm, tokenizer=tokenizer,
        num_boxes=num_boxes, num_tools=num_tools,
    ).to(DEVICE)

    # 加载离线预训练 Router 权重（如果存在）
    off_ckpt = task_cfg.get("offline_router_path")
    if off_ckpt and os.path.exists(off_ckpt):
        if accelerator.is_main_process:
            print(f">> 注入离线 Router: {off_ckpt}")
        off_router = torch.load(off_ckpt, map_location=DEVICE)
        if ablation == "linear_router":
            if accelerator.is_main_process:
                print(f">> [E4] 跳过离线 Router 权重加载（SimpleLinearRouter 结构不同）")
        else:
            router.load_state_dict(off_router['router_state_dict'])
        llm_caller.query_proj.load_state_dict(off_router['query_proj_state_dict'])

    # --- Router 半冻结 ---
    if ablation == "linear_router":
        router.train()
        if accelerator.is_main_process:
            print(f">> [E4 全参数微调] SimpleLinearRouter 所有参数参与训练")
    else:
        router.eval()
        for param in router.parameters():
            param.requires_grad = False

    llm_caller.query_proj.train()
    for param in llm_caller.query_proj.parameters():
        param.requires_grad = False

    # --- 损失函数（使用 AblationLoss）---
    geo_off = getattr(args, 'geo_loss_off', False) or (ablation == "geo_loss_off")
    loss_fn = create_ablation_loss(geo_off=geo_off)
    if accelerator.is_main_process and geo_off:
        print(f">> [E1 无几何损失] geo_loss_off=True，几何包含损失已关闭")

    from data_process.dataset import get_dataloader as get_base_dataloader
    dataloader = get_base_dataloader(batch_size=task_cfg["batch_size"])

    trainer = BaseTrainerNew(
        router=router, llm_caller=llm_caller,
        loss_fn=loss_fn, dataloader=dataloader,
        device=DEVICE, accelerator=accelerator,
    )

    os.makedirs(ckpt_dir, exist_ok=True)

    history_total = []
    best_loss = float('inf')
    best_epoch = -1
    patience = 5
    no_improve = 0

    start_epoch = 0
    if resume:
        start_epoch = trainer.load_checkpoint(resume) + 1
        if accelerator.is_main_process:
            print(f">> 🔄 从 epoch={start_epoch} 继续训练")

    for epoch in range(start_epoch, task_cfg["epochs"]):
        if accelerator.is_main_process:
            print(f"\n--- Epoch {epoch} ---")
        t_loss, task_loss, geo_loss, cont_loss = trainer.train_epoch(epoch)

        if accelerator.is_main_process:
            history_total.append((epoch, t_loss))
            if t_loss < best_loss:
                best_loss = t_loss
                best_epoch = epoch
                no_improve = 0
                trainer.save_checkpoint(epoch, save_dir=ckpt_dir, suffix="_best")
                print(f"🏆 Epoch {epoch} 最佳！Loss: {t_loss:.4f}")
            else:
                no_improve += 1
                print(f"📉 Loss 未改善 ({no_improve}/{patience})，最佳: {best_loss:.4f} (Epoch {best_epoch})")

            if no_improve >= patience:
                print(f"\n🛑 早停！连续 {patience} 个 epoch 未改善。")
                break

        stop_tensor = torch.tensor(1 if no_improve >= patience else 0, device=DEVICE)
        stop_tensor = accelerator.reduce(stop_tensor, reduction="max")
        if stop_tensor.item() > 0:
            break
        accelerator.wait_for_everyone()

    if accelerator.is_main_process and history_total:
        plt.figure(figsize=(10, 6))
        epochs_list, losses_list = zip(*history_total)
        plt.plot(epochs_list, losses_list, label='Total Loss', marker='o', linewidth=2)
        plt.title(f'{task_cfg["name"]} Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.grid(True)
        plt.savefig("ablation_base_loss_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Loss 曲线已保存为: ablation_base_loss_curve.png")

    if accelerator.is_main_process:
        print(f"\n✅ Base 训练完成！最佳: Epoch {best_epoch}, Loss: {best_loss:.4f}")
        print(f"   权重保存在: {ckpt_dir}")


# ============================================================================
# Mode: train (CL Task1/2/3)
# ============================================================================


def run_train_cl(task_cfg, prev_task_cfg, args, resume=None):
    ablation = getattr(args, 'ablation', None)
    ckpt_dir = get_checkpoint_dir(task_cfg["name"], ablation)

    print(f"\n{'='*50}")
    print(f"[Train] {task_cfg['name']} — 持续学习训练（CL 版）")
    if ablation:
        print(f"[Train] 消融模式: {ablation}")
    print(f"[Train] 保存目录: {ckpt_dir}")
    print(f"{'='*50}")

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(gradient_accumulation_steps=2, kwargs_handlers=[ddp_kwargs])
    DEVICE = accelerator.device

    if accelerator.is_main_process:
        print(f">> 设备: {DEVICE}")

    # --- 加载物理空间 ---
    l2_centers = torch.load(task_cfg["l2_centers"],  map_location=DEVICE, weights_only=False).to(torch.bfloat16)
    l2_widths  = torch.load(task_cfg["l2_widths"],   map_location=DEVICE, weights_only=False).to(torch.bfloat16)
    l1_centers = torch.load(task_cfg["l1_centers"],  map_location=DEVICE, weights_only=False).to(torch.bfloat16)
    l1_widths  = torch.load(task_cfg["l1_widths"],   map_location=DEVICE, weights_only=False).to(torch.bfloat16)
    m_global   = torch.load(task_cfg["m_global"],    map_location=DEVICE, weights_only=False).to(torch.bfloat16)

    total_num_boxes = l2_centers.shape[0]
    total_num_tools = m_global.shape[0]
    dim = l2_centers.shape[1]

    if accelerator.is_main_process:
        print(f">> 物理空间: {total_num_boxes} 盒子, {total_num_tools} 工具")

    # --- Router（使用工厂函数）---
    router_type = ABLATION_TO_ROUTER_TYPE.get(ablation, "dual_stream")
    if ablation == "linear_router":
        router = create_router(
            router_type="simple_linear",
            dim=dim, num_tools=total_num_tools, num_boxes=total_num_boxes,
        ).to(DEVICE)
        if accelerator.is_main_process:
            print(f">> [E4 线性路由器] 使用 SimpleLinearRouter（无 box 机制）")
    elif ablation in ("w/o_hierarchy", "flat_space"):
        router = create_router(
            router_type="flat",
            dim=dim, num_tools=total_num_tools, num_boxes=total_num_boxes,
            m_global_tool=m_global,
            l2_centers=l2_centers, l2_widths=l2_widths,
        ).to(DEVICE)
        if accelerator.is_main_process:
            print(f">> [扁平空间] 使用 FlatDualStreamRouter（L1 层级已移除）")
    else:
        router = create_router(
            router_type=router_type,
            dim=dim, num_tools=total_num_tools, num_boxes=total_num_boxes,
            m_global=m_global,
            l2_centers=l2_centers, l2_widths=l2_widths,
            l1_centers=l1_centers, l1_widths=l1_widths,
        ).to(DEVICE)

    # --- 半冻结策略 ---
    semi_freeze_off = (getattr(args, 'ablation', None) == "semi_freeze_off")
    if ablation == "linear_router":
        # SimpleLinearRouter: 所有参数可训练（无 l2_centers 需要解冻）
        router.train()
        if accelerator.is_main_process:
            print(f">> [E4 全参数微调] SimpleLinearRouter 所有参数参与训练")
    elif semi_freeze_off:
        if accelerator.is_main_process:
            print(f">> [D2 无半冻结] 全参数解冻，所有参数参与训练")
        for param in router.parameters():
            param.requires_grad = True
    else:
        if accelerator.is_main_process:
            print(f">> 物理半冻结：仅解冻新盒子 l2_centers 坐标")
        for name, param in router.named_parameters():
            if name == "l2_centers":
                param.requires_grad = True
            else:
                param.requires_grad = False

    # --- 加载 Llama ---
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    base_llm = AutoModelForCausalLM.from_pretrained(LLAMA_PATH, torch_dtype=torch.bfloat16).to(DEVICE)
    base_llm.eval()
    for param in base_llm.parameters():
        param.requires_grad = False

    # --- 权重继承 ---
    weight_inherit_off = getattr(args, 'weight_inherit_off', False)
    old_ckpt = None
    if weight_inherit_off:
        real_old_num_boxes = prev_task_cfg["old_num_boxes"]
        real_old_num_tools = prev_task_cfg["old_num_tools"]
        if accelerator.is_main_process:
            print(f">> [D3 无权重继承] 跳过旧 checkpoint，使用随机初始化！")
    else:
        old_ckpt_path = args.old_checkpoint if hasattr(args, 'old_checkpoint') and args.old_checkpoint else task_cfg.get("old_checkpoint")
        if old_ckpt_path and os.path.exists(old_ckpt_path):
            if accelerator.is_main_process:
                print(f">> [权重继承] 注入旧任务记忆: {old_ckpt_path}")
            old_ckpt = torch.load(old_ckpt_path, map_location=DEVICE)
            real_old_num_boxes = old_ckpt['prompt_pool'].shape[0]
            real_old_num_tools = old_ckpt['classifier_state_dict']['weight'].shape[0]
            if 'old_num_tools' in old_ckpt:
                if old_ckpt['old_num_tools'] == old_ckpt['classifier_state_dict']['weight'].shape[0]:
                    real_old_num_tools = old_ckpt['old_num_tools']
            if 'old_num_boxes' in old_ckpt:
                if old_ckpt['old_num_boxes'] == old_ckpt['prompt_pool'].shape[0]:
                    real_old_num_boxes = old_ckpt['old_num_boxes']
            if real_old_num_boxes >= total_num_boxes and prev_task_cfg is not None:
                real_old_num_boxes = prev_task_cfg.get("old_num_boxes", real_old_num_boxes)
                real_old_num_tools = prev_task_cfg.get("old_num_tools", real_old_num_tools)
            if accelerator.is_main_process:
                print(f">> checkpoint 物理空间: {real_old_num_boxes} 盒子, {real_old_num_tools} 工具")
        else:
            real_old_num_boxes = prev_task_cfg["old_num_boxes"]
            real_old_num_tools = prev_task_cfg["old_num_tools"]
            if accelerator.is_main_process:
                print(f"⚠️ 找不到旧权重: {old_ckpt_path}，使用 fallback 值")

    # --- LLM Caller（使用工厂函数）---
    llm_caller = create_llm_caller(
        caller_type="cl",
        base_llm=base_llm, tokenizer=tokenizer,
        num_boxes=total_num_boxes, num_tools=total_num_tools,
        old_num_boxes=real_old_num_boxes, old_num_tools=real_old_num_tools,
    ).to(DEVICE)

    if old_ckpt is not None and ablation == "linear_router":
        # SimpleLinearRouter: 无 l2_centers，跳过 router 权重继承
        llm_caller.query_proj.load_state_dict(old_ckpt['query_proj_state_dict'])
        llm_caller.query_proj.eval()
        for param in llm_caller.query_proj.parameters():
            param.requires_grad = False
        llm_caller.load_base_weights(
            old_prompt_tensor=old_ckpt['prompt_pool'][:real_old_num_boxes],
            old_classifier_state_dict=old_ckpt['classifier_state_dict']
        )
        if accelerator.is_main_process:
            print(f">> [E4 线性路由器] 从 ckpt 继承 query_proj 和 prompt_pool，Router 使用随机初始化")
    elif old_ckpt is not None and ablation != "w/o_hierarchy":
        old_router_state = old_ckpt['router_state_dict']
        with torch.no_grad():
            router.l2_centers.data[:real_old_num_boxes] = old_router_state['l2_centers'][:real_old_num_boxes].to(DEVICE)

        llm_caller.query_proj.load_state_dict(old_ckpt['query_proj_state_dict'])
        llm_caller.query_proj.eval()
        for param in llm_caller.query_proj.parameters():
            param.requires_grad = False

        llm_caller.load_base_weights(
            old_prompt_tensor=old_ckpt['prompt_pool'][:real_old_num_boxes],
            old_classifier_state_dict=old_ckpt['classifier_state_dict']
        )
    elif old_ckpt is not None and ablation == "w/o_hierarchy":
        old_router_state = old_ckpt['router_state_dict']
        with torch.no_grad():
            router.l2_centers.data[:real_old_num_boxes] = old_router_state['l2_centers'][:real_old_num_boxes].to(DEVICE)
        llm_caller.query_proj.load_state_dict(old_ckpt['query_proj_state_dict'])
        llm_caller.query_proj.eval()
        for param in llm_caller.query_proj.parameters():
            param.requires_grad = False
        llm_caller.load_base_weights(
            old_prompt_tensor=old_ckpt['prompt_pool'][:real_old_num_boxes],
            old_classifier_state_dict=old_ckpt['classifier_state_dict']
        )
        if accelerator.is_main_process:
            print(f">> [扁平空间] 从 hierarchical ckpt 继承 {real_old_num_boxes} 个 l2_centers")

    # --- semi_freeze_off 全解冻：解冻老分类器、老 Prompt、query_proj ---
    if semi_freeze_off:
        if accelerator.is_main_process:
            print(f">> [D2 无物理半冻结] 解冻老分类器、老 Prompt Pool、query_proj！")
        # query_proj
        for param in llm_caller.query_proj.parameters():
            param.requires_grad = True
        # 老分类器
        for param in llm_caller.classifier_old.parameters():
            param.requires_grad = True
        # 老 Prompt Pool
        llm_caller.prompt_pool_old.requires_grad = True

    # --- 经验回放 ---
    replay_off = getattr(args, 'replay_off', False)
    replay_per_tool = getattr(args, 'replay_per_tool', 1)
    dataloader_kwargs = dict(
        task=task_cfg["name"].lower(),
        batch_size=task_cfg["batch_size"],
        train_json=task_cfg.get("train_json"),
        new_tools_json=task_cfg.get("new_tools_json"),
        tool_to_l2_path=task_cfg.get("tool_to_l2"),
        l2_to_l1_path=task_cfg.get("l2_to_l1"),
        prev_task_tools_json=task_cfg.get("prev_task_tools_json"),
        prev_task_train_json=task_cfg.get("prev_task_train_json"),
    )
    if replay_off or replay_per_tool == 0:
        if accelerator.is_main_process:
            print(f">> [D4 无经验回放] 关闭 replay，训练集仅包含当前任务数据")
        dataloader_kwargs["replay_per_tool"] = 0
    elif replay_per_tool != 1:
        dataloader_kwargs["replay_per_tool"] = replay_per_tool

    dataloader = build_dataloader(**dataloader_kwargs)

    # --- 损失函数（使用 create_ablation_loss）---
    geo_off = getattr(args, 'geo_loss_off', False) or (ablation == "geo_loss_off")
    contrast_off = getattr(args, 'contrast_loss_off', False)
    loss_fn = create_ablation_loss(geo_off=geo_off, contrast_off=contrast_off)
    if accelerator.is_main_process:
        if geo_off:
            print(f">> [E1 无几何损失] geo_loss_off=True，几何包含损失已关闭")
        if contrast_off:
            print(f">> [B3 无对比损失] contrast_loss_off=True，路由对比损失已关闭")

    trainer = CLTrainerNew(
        router=router, llm_caller=llm_caller,
        loss_fn=loss_fn, dataloader=dataloader,
        device=DEVICE, accelerator=accelerator,
        ablation_cfg={"semi_freeze_off": semi_freeze_off}
    )

    # NaN 保护逻辑已内置在 CLTrainerNew.train_epoch 中

    os.makedirs(ckpt_dir, exist_ok=True)

    history_total, history_task, history_geo, history_cont = [], [], [], []
    start_epoch = 0

    if resume:
        start_epoch = trainer.load_checkpoint(resume) + 1
        if accelerator.is_main_process:
            print(f">> 🔄 从 epoch={start_epoch} 继续训练")
    elif os.path.exists(ckpt_dir):
        latest = _get_latest_checkpoint(ckpt_dir)
        if latest:
            start_epoch = trainer.load_checkpoint(os.path.join(ckpt_dir, latest)) + 1
            if accelerator.is_main_process:
                print(f">> 🔄 自动检测到 checkpoint，从 epoch={start_epoch} 继续训练")

    for epoch in range(start_epoch, task_cfg["epochs"]):
        if accelerator.is_main_process:
            print(f"\n--- Epoch {epoch} ---")
        t_loss, task_loss, geo_loss, cont_loss, early_stop, is_best = trainer.train_epoch(epoch)

        if accelerator.is_main_process:
            history_total.append(t_loss)
            history_task.append(task_loss)
            history_geo.append(geo_loss)
            history_cont.append(cont_loss)

        stop_tensor = torch.tensor(1 if early_stop else 0, device=DEVICE)
        stop_tensor = accelerator.reduce(stop_tensor, reduction="max")

        if accelerator.is_main_process and is_best:
            trainer.save_checkpoint(epoch, save_dir=ckpt_dir)

        if stop_tensor.item() > 0:
            if accelerator.is_main_process:
                print(f"🛑 早停，所有 GPU 退出训练循环！")
            break
        accelerator.wait_for_everyone()

    if accelerator.is_main_process and history_total:
        task_name_lower = task_cfg["name"].lower().replace(" ", "_")
        ab_tag = f"_{ablation}" if ablation else ""
        png_filename = f"{task_name_lower}{ab_tag}_loss_curve.png"
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(history_total) + 1), history_total, label='Total Loss', marker='o', linewidth=2)
        plt.plot(range(1, len(history_task) + 1), history_task, label='Task Loss', linestyle='--')
        plt.plot(range(1, len(history_geo) + 1), history_geo, label='Geo Loss', linestyle=':')
        plt.plot(range(1, len(history_cont) + 1), history_cont, label='Cont Loss', linestyle='-.')
        plt.title(f'{task_cfg["name"]} Continual Learning Loss Curve{ab_tag}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(png_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Loss 曲线已保存为: {png_filename}")

    if accelerator.is_main_process:
        print(f"\n✅ {task_cfg['name']} 训练完成！权重保存在: {ckpt_dir}")


# ============================================================================
# Mode: eval
# ============================================================================
def run_eval(model_task_cfg, eval_task_cfg, args, limit):
    print(f"\n{'='*50}")
    print(f"[Eval] 用 {model_task_cfg['name']} 的权重，在 {eval_task_cfg['name']} 的测试集上评估")
    if args.ablation:
        print(f"[Eval] 消融模式: {args.ablation}")
    print(f"{'='*50}")

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    l2_centers = torch.load(model_task_cfg["l2_centers"],  map_location=DEVICE, weights_only=False).to(torch.bfloat16)
    l2_widths  = torch.load(model_task_cfg["l2_widths"],   map_location=DEVICE, weights_only=False).to(torch.bfloat16)
    l1_centers = torch.load(model_task_cfg["l1_centers"],  map_location=DEVICE, weights_only=False).to(torch.bfloat16)
    l1_widths  = torch.load(model_task_cfg["l1_widths"],   map_location=DEVICE, weights_only=False).to(torch.bfloat16)
    m_global   = torch.load(model_task_cfg["m_global"],    map_location=DEVICE, weights_only=False).to(torch.bfloat16)
    tool_to_l2 = torch.load(model_task_cfg["tool_to_l2"],  map_location="cpu", weights_only=False)

    total_num_boxes = l2_centers.shape[0]
    total_num_tools = m_global.shape[0]
    dim = l2_centers.shape[1]

    # --- 加载 checkpoint ---
    ablation = getattr(args, 'ablation', None)
    ckpt_dir = get_checkpoint_dir(model_task_cfg["name"], ablation)
    ckpt_files = []
    if os.path.exists(ckpt_dir):
        ckpt_files = sorted(
            [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")],
            key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x))
        )

    if not ckpt_files:
        print(f"❌ 未找到 checkpoint: {ckpt_dir}")
        return

    ckpt_path = os.path.join(ckpt_dir, ckpt_files[-1])
    mtime_str = datetime.datetime.fromtimestamp(os.path.getmtime(ckpt_path)).strftime("%Y-%m-%d %H:%M:%S")
    print(f">> 加载 checkpoint: {ckpt_path}  ({mtime_str})")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    ckpt_router_state = checkpoint['router_state_dict']

    has_l2_centers = 'l2_centers' in ckpt_router_state

    if has_l2_centers:
        ckpt_l2 = ckpt_router_state['l2_centers']
        ckpt_num_boxes = ckpt_l2.shape[0]
        dim = ckpt_l2.shape[1]
    else:
        # SimpleLinearRouter: 从 prompt_proj 权重推断 dim
        prompt_proj_weight = ckpt_router_state.get('prompt_proj.weight')
        if prompt_proj_weight is not None:
            dim = prompt_proj_weight.shape[1]
        else:
            dim = 4096  # fallback
        ckpt_num_boxes = checkpoint['prompt_pool'].shape[0]

    cls_state = checkpoint['classifier_state_dict']
    if 'weight_old' in cls_state:
        # CL 格式：old/new 分离，保存为 weight_old/bias_old/weight_new/bias_new
        ckpt_old_num_tools = cls_state['weight_old'].shape[0]
        ckpt_new_num_tools = cls_state['weight_new'].shape[0]
        ckpt_num_tools = ckpt_old_num_tools + ckpt_new_num_tools
        is_cl_format = True
        # prompt_pool 分割点：从 checkpoint 的 old_num_boxes 读取
        ckpt_old_num_boxes = checkpoint.get('old_num_boxes', checkpoint['prompt_pool'].shape[0])
    else:
        # 基线格式：单一 weight/bias
        ckpt_num_tools = cls_state['weight'].shape[0]
        is_cl_format = False

    if has_l2_centers:
        l2_centers_use = ckpt_l2.detach().clone().to(torch.bfloat16)
        l2_widths_use  = ckpt_router_state['l2_widths'].detach().clone().to(torch.bfloat16)
    else:
        l2_centers_use = l2_centers[:ckpt_num_boxes]
        l2_widths_use  = l2_widths[:ckpt_num_boxes]

    m_global_use = m_global[:ckpt_num_tools, :ckpt_num_boxes]

    from models.router import DualStreamRouter
    from models.llm_caller import LLMCaller

    if ablation == "linear_router":
        from models.router import SimpleLinearRouter
        router = SimpleLinearRouter(
            dim=dim, num_tools=ckpt_num_tools, num_boxes=ckpt_num_boxes
        ).to(DEVICE)
        print(f">> [E4 线性路由器] 使用 SimpleLinearRouter 评估")
    elif ablation in ("router_semantic", "router_dependency", "router_no_gate"):
        base_router = DualStreamRouter(
            dim=dim, num_tools=ckpt_num_tools, num_boxes=ckpt_num_boxes,
            m_global=m_global_use,
            l2_centers=l2_centers_use, l2_widths=l2_widths_use,
            l1_centers=l1_centers, l1_widths=l1_widths
        ).to(DEVICE)
        router = AblationRouter(base_router, ablation_mode=ablation)
        print(f">> [推理消融] 使用 AblationRouter(ablation_mode={ablation})")
    elif ablation == "w/o_hierarchy":
        from models.router import FlatDualStreamRouter
        router = FlatDualStreamRouter(
            dim=dim, num_tools=ckpt_num_tools, num_boxes=ckpt_num_boxes,
            m_global_tool=m_global_use,
            l2_centers=l2_centers_use, l2_widths=l2_widths_use,
        ).to(DEVICE)
    else:
        router = DualStreamRouter(
            dim=dim, num_tools=ckpt_num_tools, num_boxes=ckpt_num_boxes,
            m_global=m_global_use,
            l2_centers=l2_centers_use, l2_widths=l2_widths_use,
            l1_centers=l1_centers, l1_widths=l1_widths
        ).to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    base_llm = AutoModelForCausalLM.from_pretrained(LLAMA_PATH, torch_dtype=torch.bfloat16).to(DEVICE)

    # CL 格式用 CLCaller（old/new 分离），基线格式用 LLMCaller
    if is_cl_format:
        from models.llm_caller_cl import LLMCaller as CLCaller
        llm_caller = CLCaller(
            base_llm=base_llm, tokenizer=tokenizer,
            num_boxes=ckpt_num_boxes, num_tools=ckpt_num_tools,
            old_num_boxes=ckpt_old_num_boxes, old_num_tools=ckpt_old_num_tools
        ).to(DEVICE)
    else:
        from models.llm_caller import LLMCaller
        llm_caller = LLMCaller(
            base_llm=base_llm, tokenizer=tokenizer,
            num_boxes=ckpt_num_boxes, num_tools=ckpt_num_tools
        ).to(DEVICE)

    router.load_state_dict(ckpt_router_state)

    llm_caller_state = {}
    for k, v in checkpoint.get('query_proj_state_dict', {}).items():
        llm_caller_state[f'query_proj.{k}'] = v

    if is_cl_format:
        # CL 格式：weight_old/bias_old/weight_new/bias_new
        llm_caller_state['classifier_old.weight'] = cls_state['weight_old'].to(DEVICE)
        llm_caller_state['classifier_old.bias']   = cls_state['bias_old'].to(DEVICE)
        llm_caller_state['classifier_new.weight'] = cls_state['weight_new'].to(DEVICE)
        llm_caller_state['classifier_new.bias']   = cls_state['bias_new'].to(DEVICE)
        prompt_pool = checkpoint.get('prompt_pool')
        if prompt_pool is not None:
            llm_caller_state['prompt_pool_old'] = prompt_pool[:ckpt_old_num_boxes].to(DEVICE)
    else:
        for k, v in cls_state.items():
            llm_caller_state[f'classifier.{k}'] = v
        if 'prompt_pool' in checkpoint:
            llm_caller_state['prompt_pool'] = checkpoint['prompt_pool'].to(DEVICE)

    llm_caller.load_state_dict(llm_caller_state, strict=False)

    router.eval()
    llm_caller.eval()

    name_to_id = load_data_mappings(eval_task_cfg)

    eval_json = eval_task_cfg["eval_json"]
    if not os.path.exists(eval_json):
        print(f"❌ 测试文件不存在: {eval_json}")
        return

    with open(eval_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(tool_to_l2, dict):
        tool_to_l2_map = {tid: bid for tid, bid in tool_to_l2.items() if bid < ckpt_num_boxes}
    else:
        tool_to_l2_map = tool_to_l2[:ckpt_num_boxes]

    valid_samples = 0
    tool_recall_1, tool_recall_3, tool_recall_5 = 0, 0, 0
    tool_ndcg_1, tool_ndcg_3, tool_ndcg_5 = 0.0, 0.0, 0.0

    data_to_eval = data if limit <= 0 else data[:limit]
    print(f"\n开始评估（{len(data_to_eval)} 条）...")

    pbar = tqdm(data_to_eval, desc=f"评估 {eval_task_cfg['name']}", unit="条", ncols=80)
    for item in pbar:
        conversations = item.get("conversations", [])
        query_text, target_str = "", ""
        for conv in conversations:
            if conv.get("role") == "user":
                query_text = conv.get("content", "").strip()
            elif conv.get("role") == "assistant":
                target_str = conv.get("content", "").strip()

        if not query_text or not target_str:
            continue

        match = re.match(r'<<(.+?)&&(.+?)>>', target_str)
        if not match:
            continue

        target_norm = normalize(f"{match.group(2).strip()}for{match.group(1).strip()}")
        if target_norm not in name_to_id:
            continue

        target_tool_id = name_to_id[target_norm]
        if target_tool_id not in tool_to_l2_map:
            continue

        if valid_samples <= 5:
            print(f"  [DEBUG target] target_norm={target_norm!r}  target_tool_id={target_tool_id}  "
                  f"in_tool_to_l2={target_tool_id in tool_to_l2_map}  "
                  f"tool_to_l2_map[target]={tool_to_l2_map.get(target_tool_id, 'N/A')}")

        valid_samples += 1

        with torch.no_grad():
            q_vec, input_ids, attention_mask = llm_caller.extract_query_vector([query_text])
            t_prev = torch.tensor([-1], dtype=torch.long, device=DEVICE)
            s_total, _ = router(q_vec, t_prev)
            pred_l2_id = s_total.argmax(dim=-1)
            logits = llm_caller(input_ids, attention_mask, pred_l2_id)

            # --- debug: 前 5 条打印中间值 ---
            if valid_samples <= 5:
                print(f"\n  [DEBUG sample] q_vec={q_vec.shape} mean={q_vec.mean().item():.4f} std={q_vec.std().item():.4f}  "
                      f"s_total={s_total.shape} range=[{s_total.min().item():.4f},{s_total.max().item():.4f}]  "
                      f"pred_l2={pred_l2_id.item()}  logits={logits.shape} range=[{logits.min().item():.4f},{logits.max().item():.4f}]")

            num_logits = logits.shape[-1]
            _, top_k_tool_ids = torch.topk(logits, k=min(5, num_logits), dim=-1)
            top_k_tools_list = top_k_tool_ids[0].tolist()

            if target_tool_id >= num_logits:
                continue

            if valid_samples <= 5:
                print(f"  [DEBUG top5] target={target_tool_id}  top5={top_k_tools_list}  "
                      f"hit@1={target_tool_id in top_k_tools_list[:1]}")

        if target_tool_id in top_k_tools_list[:1]:
            tool_recall_1 += 1
            rank = top_k_tools_list.index(target_tool_id) + 1
            tool_ndcg_1 += 1.0 / math.log2(rank + 1)

        if target_tool_id in top_k_tools_list[:3]:
            tool_recall_3 += 1
            rank = top_k_tools_list.index(target_tool_id) + 1
            tool_ndcg_3 += 1.0 / math.log2(rank + 1)

        if target_tool_id in top_k_tools_list[:5]:
            tool_recall_5 += 1
            rank = top_k_tools_list.index(target_tool_id) + 1
            tool_ndcg_5 += 1.0 / math.log2(rank + 1)

        pbar.set_postfix_str(f"R@1={tool_recall_1/valid_samples*100:.1f}%" if valid_samples > 0 else "")

    print(f"\n{'='*50}")
    print(f"🎯 评估完成！共测试 {valid_samples} 条数据。")
    if valid_samples > 0:
        print(f"🛠️  Recall@1: {tool_recall_1 / valid_samples * 100:.2f}%")
        print(f"🛠️  Recall@3: {tool_recall_3 / valid_samples * 100:.2f}%")
        print(f"🛠️  Recall@5: {tool_recall_5 / valid_samples * 100:.2f}%")
        print(f"🏆  NDCG@1:   {tool_ndcg_1 / valid_samples * 100:.2f}%")
        print(f"🏆  NDCG@3:   {tool_ndcg_3 / valid_samples * 100:.2f}%")
        print(f"🏆  NDCG@5:   {tool_ndcg_5 / valid_samples * 100:.2f}%")
    print(f"{'='*50}")

    return {
        "recall@1": tool_recall_1 / valid_samples * 100 if valid_samples > 0 else 0.0,
        "recall@3": tool_recall_3 / valid_samples * 100 if valid_samples > 0 else 0.0,
        "recall@5": tool_recall_5 / valid_samples * 100 if valid_samples > 0 else 0.0,
        "ndcg@1":   tool_ndcg_1 / valid_samples * 100 if valid_samples > 0 else 0.0,
        "ndcg@3":   tool_ndcg_3 / valid_samples * 100 if valid_samples > 0 else 0.0,
        "ndcg@5":   tool_ndcg_5 / valid_samples * 100 if valid_samples > 0 else 0.0,
    }


# ============================================================================
# Mode: ablation_compare（推理时消融综合对比）
# ============================================================================
INFERENCE_ABLATIONS = {"router_semantic", "router_dependency", "router_no_gate"}


def run_ablation_compare(task_name, args, limit):
    """推理时消融综合对比——用基线 checkpoint，依次评估不同消融模式"""
    ablation = getattr(args, 'ablation', None)
    if ablation != "ablation_compare":
        return

    print(f"\n{'#'*60}")
    print(f"## 推理时消融综合对比 | 模型={task_name}")
    print(f"{'#'*60}")

    if task_name not in TASK_CONFIGS:
        print(f"❌ 未知任务: {task_name}")
        return

    model_cfg = TASK_CONFIGS[task_name]
    eval_cfg = model_cfg

    results_table = []

    # 基线
    result_base = run_eval(model_cfg, eval_cfg, args, limit)
    if result_base:
        results_table.append({"模型": task_name, "评估集": task_name, "消融": "基线（完整）", **result_base})

    # 推理时消融
    for ab in sorted(INFERENCE_ABLATIONS):
        args_with_ab = argparse.Namespace(
            ablation=ab,
            old_checkpoint=getattr(args, 'old_checkpoint', None),
            limit=getattr(args, 'limit', -1),
        )
        result = run_eval(model_cfg, eval_cfg, args_with_ab, limit)
        if result:
            results_table.append({"模型": task_name, "评估集": task_name, "消融": ab, **result})

    # 打印汇总
    print(f"\n{'='*80}")
    print(f"🎯 推理时消融对比结果 | 模型={task_name}")
    print(f"{'='*80}")
    header = f"{'消融项':<30} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'N@5':>8}"
    print(header)
    print("-" * 70)
    for r in results_table:
        print(f"{r['消融']:<30} "
              f"{r.get('recall@1', 0.0):>7.2f}% {r.get('recall@3', 0.0):>7.2f}% "
              f"{r.get('recall@5', 0.0):>7.2f}% {r.get('ndcg@5', 0.0):>7.2f}%")

    result_file = f"./ablation_compare_results_{task_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results_table, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 结果已保存到: {result_file}")


# ============================================================================
# 主入口
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="IH-PromptDSI 消融实验统一入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：

  # 基线训练（无需 --ablation）
  python run_ablation.py --task task1 --mode train --train-only

  # 消融实验：训练 + 评估
  python run_ablation.py --task task1 --mode all --ablation semi_freeze_off
  python run_ablation.py --task task1 --mode all --ablation geo_loss_off
  python run_ablation.py --task task1 --mode all --ablation weight_inherit_off
  python run_ablation.py --task task1 --mode all --ablation replay_off
  python run_ablation.py --task task1 --mode all --ablation w/o_hierarchy
  python run_ablation.py --task task1 --mode all --ablation flat_space

  # 推理时消融（用现有 checkpoint 评估，无需训练）
  python run_ablation.py --task task1 --mode eval --ablation router_semantic
  python run_ablation.py --task task1 --mode eval --ablation ablation_compare
        """
    )
    # ---------- 基础参数（与 run.py 一致） ----------
    parser.add_argument(
        "--task", default="all",
        choices=["base", "task1", "task2", "task3", "all"],
        help="目标任务（默认 all）"
    )
    parser.add_argument(
        "--mode", default="all",
        choices=["prepare", "train", "eval", "all"],
        help="运行模式（默认 all）：prepare / train / eval / all"
    )
    parser.add_argument(
        "--stage", type=int, default=3, choices=[0, 1, 2, 3],
        help="prepare 阶段（默认 3）"
    )
    parser.add_argument(
        "--limit", type=int, default=-1,
        help="评估样本上限（-1=全量，默认 -1）"
    )
    parser.add_argument(
        "--eval-tasks", type=str, default=None,
        help="评估时指定任务列表，逗号分隔"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=None
    )
    parser.add_argument(
        "--old-checkpoint", type=str, default=None,
        help="CL 训练时注入旧任务记忆的 checkpoint 路径"
    )
    parser.add_argument(
        "--gpu", type=str, default=None,
        help="指定 GPU，如 '0,1,2,3'"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="从指定 checkpoint 继续训练"
    )
    # ---------- 消融实验参数 ----------
    parser.add_argument(
        "--ablation", type=str, default=None,
        choices=[
            None,
            "semi_freeze_off",
            "weight_inherit_off",
            "replay_off",
            "geo_loss_off",
            "w/o_hierarchy",
            "flat_space",
            "linear_router",
            "router_semantic",
            "router_dependency",
            "router_no_gate",
            "ablation_compare",
        ],
        help="消融实验名称（None=基线；推理时消融：router_*；训练时消融：*_off / w/o_* / linear_router）"
    )
    # ---------- 工作流控制 flag ----------
    parser.add_argument(
        "--train-only", action="store_true",
        help="仅训练，跳过评估（训练完成后直接退出）"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="仅评估，跳过训练"
    )
    # ---------- 消融实验开关（与 --ablation 等价，支持独立指定） ----------
    parser.add_argument(
        "--geo-loss-off", action="store_true",
        help="关闭几何包含损失 L_geo（需重新训练）"
    )
    parser.add_argument(
        "--contrast-loss-off", action="store_true",
        help="关闭路由对比损失（需重新训练）"
    )
    parser.add_argument(
        "--semi-freeze-off", action="store_true",
        help="关闭物理半冻结，全参数微调（需重新训练）"
    )
    parser.add_argument(
        "--weight-inherit-off", action="store_true",
        help="关闭权重继承，随机初始化（需重新训练）"
    )
    parser.add_argument(
        "--replay-off", action="store_true",
        help="关闭经验回放（需重新训练）"
    )
    parser.add_argument(
        "--replay-per-tool", type=int, default=1,
        help="经验回放每工具采样数（默认1，设为0则关闭回放）"
    )
    parser.add_argument(
        "--space-type", default="hierarchical",
        choices=["hierarchical", "flat"],
        help="空间类型：hierarchical（默认L1/L2层级）/ flat（单层扁平）"
    )

    args = parser.parse_args()

    # 同步 --ablation 与独立 flag
    _flag_map = {
        "semi_freeze_off":    "semi_freeze_off",
        "weight_inherit_off": "weight_inherit_off",
        "replay_off":         "replay_off",
        "geo_loss_off":       "geo_loss_off",
        "w/o_hierarchy":      None,
        "flat_space":         None,
        "linear_router":      None,
        "router_semantic":    None,
        "router_dependency":  None,
        "router_no_gate":     None,
        "ablation_compare":   None,
    }
    if args.ablation:
        flag = _flag_map.get(args.ablation)
        if flag:
            setattr(args, flag, True)
        if args.ablation == "flat_space":
            args.space_type = "flat"

    # 判断任务列表
    if args.task == "all":
        task_names = ["base", "task1", "task2", "task3"]
    else:
        task_names = [args.task]

    # 判断是否需要训练（任何 --ablation 都不需要，除非是推理时消融）
    needs_train = args.mode in ("train", "all") and not args.eval_only
    is_inference_only = (
        args.ablation in INFERENCE_ABLATIONS or
        args.ablation == "ablation_compare"
    )

    # ---------- ablation_compare 单独处理 ----------
    if args.ablation == "ablation_compare":
        for tn in task_names:
            run_ablation_compare(tn, args, limit=args.limit if args.limit > 0 else -1)
        return

    # ---------- 主循环 ----------
    for task_name in task_names:
        if task_name not in TASK_CONFIGS:
            print(f"❌ 未知任务: {task_name}")
            continue

        task_cfg = TASK_CONFIGS[task_name]
        prev_task_cfg = TASK_CONFIGS[task_cfg["prev_task"]] if task_cfg["prev_task"] else None

        print(f"\n\n{'#'*60}")
        print(f"## 处理任务: {task_cfg['name']}" +
              (f" | 消融: {args.ablation}" if args.ablation else " | 基线"))
        print(f"{'#'*60}")

        # --- prepare ---
        if args.mode in ("prepare", "all") and not args.eval_only:
            if task_name == "base":
                print("[跳过] Base 任务无需 prepare")
            else:
                if args.stage <= 1:
                    run_prepare(task_name, stage=1, num_gpus=args.num_gpus)
                if args.stage <= 2:
                    run_prepare(task_name, stage=2, num_gpus=args.num_gpus)
                if args.stage >= 3:
                    run_prepare(task_name, stage=3, num_gpus=args.num_gpus)

        # --- train ---
        if needs_train and not is_inference_only:
            # 检查是否已训练完成（跳过已完成）
            ckpt_dir = get_checkpoint_dir(task_cfg["name"], args.ablation)
            if os.path.exists(ckpt_dir) and not args.resume:
                if _is_training_done(ckpt_dir, task_cfg["epochs"]):
                    print(f"\n✅ 训练已完成（{ckpt_dir}），跳过训练阶段")
                else:
                    latest = _get_latest_checkpoint(ckpt_dir)
                    if latest:
                        print(f"\n🔄 检测到未完成的训练，将从 checkpoint 继续: {latest}")
                        args.resume = os.path.join(ckpt_dir, latest)

            if task_name == "base":
                run_train_base(task_cfg, args, resume=args.resume)
            else:
                run_train_cl(task_cfg, prev_task_cfg, args, resume=args.resume)

            if args.train_only:
                print("\n✅ 训练完成（--train-only 模式，跳过评估）")
                return

        # --- eval ---
        if args.mode in ("eval", "all") or is_inference_only:
            eval_limit = args.limit if args.limit > 0 else -1
            if args.eval_tasks:
                eval_task_names = [t.strip() for t in args.eval_tasks.split(",")]
                alias = {"train": "base", "base": "base",
                         "task1": "task1", "task2": "task2", "task3": "task3"}
                eval_task_names = [alias.get(t, t) for t in eval_task_names]
                eval_task_names = [t for t in eval_task_names if t in TASK_CONFIGS]
                if not eval_task_names:
                    print("⚠️  --eval-tasks 中没有合法任务")
                    return
            else:
                eval_task_names = [task_name]

            for et in eval_task_names:
                if et not in TASK_CONFIGS:
                    continue
                run_eval(task_cfg, eval_task_cfg=TASK_CONFIGS[et], args=args, limit=eval_limit)

    print(f"\n\n{'='*50}")
    print(f"✅ 全部完成！")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
