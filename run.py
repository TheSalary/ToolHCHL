#!/usr/bin/env python3
"""
run.py
======
IH-PromptDSI 统一入口脚本。

功能：
  --mode prepare  数据准备（调用 scripts/prepare_task_data.py）
  --mode train    训练（Base 或持续学习任务）
  --mode eval     评估（Recall@1/3/5 + NDCG@1/3/5）
  --mode all      完整流程（prepare -> train -> eval）

使用示例：
  # 查看帮助
  python run.py --help

注意事项：
  - Task2/Task3 在 --mode prepare 之前必须先确保 data/taskN/raw/ 目录有数据
  - --mode prepare --stage 3 时自动调用对应脚本（base → rebuild_base_clusters.py，task1/2/3 → prepare_task_data.py）
  - 训练使用 accelerate multi-GPU
"""

import argparse
import os
import sys
import torch
import warnings
import math
import re
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

from config import LLAMA_PATH, TASK_CONFIGS, get_checkpoint_dir


# ============================================================================
# 辅助函数
# ============================================================================
def normalize(name):
    return re.sub(r'[^a-zA-Z0-9]', '', str(name)).lower()


def load_data_mappings(task_cfg):
    """加载工具 ID <-> 名称的完整映射（支持新旧工具）。"""
    name_to_id = {}

    # Base 工具字典
    base_tools_path = "./data/train/raw/train_tools_with_id.json"
    if os.path.exists(base_tools_path):
        with open(base_tools_path, "r", encoding="utf-8") as f:
            base_tools = json.load(f)
        for t in base_tools:
            api_norm = normalize(t.get("api_name", ""))
            tool_norm = normalize(t.get("tool_name", ""))
            name_to_id[f"{api_norm}for{tool_norm}"] = t.get("tool_id")

    # 新任务工具字典（如果存在）
    task_name = task_cfg["name"].lower()
    for prefix in ["task1", "task2", "task3"]:
        if prefix in task_name or task_cfg["cluster_dir"].startswith(f"./data/{prefix}"):
            for n in ["task1", "task2", "task3"]:
                cluster_dir = f"./data/{n}/clusters"
                tools_json = os.path.join(cluster_dir, f"{n}_tools_with_id.json")
                if os.path.exists(tools_json):
                    with open(tools_json, "r", encoding="utf-8") as f:
                        new_tools = json.load(f)
                    for t_id_str, text in new_tools.items():
                        if isinstance(text, str):
                            a_match = re.search(r'API:\s*(.*?)\.(?:\s*API Description:|$)', text)
                            t_match = re.search(r'Tool:\s*(.*?)\.(?:\s*Description:|\s*API:)', text)
                            if a_match and t_match:
                                a_name = a_match.group(1).strip()
                                t_name = t_match.group(1).strip()
                                name_to_id[normalize(f"{a_name}for{t_name}")] = int(t_id_str)
    return name_to_id


# ============================================================================
# Loss 历史记录工具
# ============================================================================

def _task_log_path(task_cfg):
    """返回 loss 日志 JSON 的保存路径。"""
    return os.path.join(task_cfg["ckpt_dir"], f"{task_cfg['name'].lower()}_loss_log.json")


def load_loss_history(task_cfg):
    """从 JSON 文件加载 loss 历史记录（用于重新绘图）。"""
    path = _task_log_path(task_cfg)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_loss_history(task_cfg, history):
    """将 loss 历史记录保存到 JSON 文件。"""
    path = _task_log_path(task_cfg)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"💾 Loss 历史已保存至: {path}")


def plot_loss_curve(history, title, png_path):
    """根据 loss 历史记录绘制曲线并保存 PNG。"""
    epochs_range = range(1, len(history["total"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history["total"],   label='Total Loss', marker='o', linewidth=2)
    plt.plot(epochs_range, history["task"],    label='Task Loss',  linestyle='--')
    plt.plot(epochs_range, history["geo"],     label='Geo Loss',   linestyle=':')
    plt.plot(epochs_range, history["cont"],    label='Cont Loss',  linestyle='-.')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✅ Loss 曲线已保存为: {png_path}")
    plt.close()


# ============================================================================
# Mode: prepare
# ============================================================================
def run_prepare(task_name, stage, num_gpus):
    """调用 scripts/prepare_task_data.py；base + stage 3 时调用 rebuild_base_clusters.py"""
    import subprocess
    print(f"\n{'='*50}")
    print(f"[Prepare] 任务={task_name} stage={stage}")
    print(f"{'='*50}")

    # base 的 stage 3 走独立流水线
    if task_name == "base" and stage == 3:
        cmd = [sys.executable, "scripts/rebuild_base_clusters.py"]
        print(f"命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=os.getcwd())
    else:
        cmd = [
            sys.executable,
            "scripts/prepare_task_data.py",
            "--task", task_name,
            "--stage", str(stage),
        ]
        print(f"命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=os.getcwd())

    if result.returncode != 0:
        print(f"❌ prepare 失败！returncode={result.returncode}")
        sys.exit(result.returncode)
    print(f"✅ prepare 完成！")


# ============================================================================
# Mode: train (Base)
# ============================================================================
def _import_base_modules():
    from models.router import DualStreamRouter
    from models.llm_caller import LLMCaller
    from training.trainer import IH_Trainer as BaseTrainer
    from training.losses import IHLoss
    from data_process.dataset import get_dataloader as get_base_dataloader
    return DualStreamRouter, LLMCaller, BaseTrainer, IHLoss, get_base_dataloader


def run_train_base(task_cfg, args=None):
    """
    Base 训练。
    """
    ablation = getattr(args, 'ablation', None) if args else None
    ckpt_dir = get_checkpoint_dir(task_cfg["name"], ablation)

    print(f"\n{'='*50}")
    print(f"[Train] {task_cfg['name']} — Base 训练")
    if ablation:
        print(f"[Train] 消融模式: {ablation}")
    print(f"[Train] 保存目录: {ckpt_dir}")
    print(f"{'='*50}")

    print(f"{'='*50}")

    DualStreamRouter, LLMCaller, BaseTrainer, IHLoss, get_base_dataloader = _import_base_modules()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(gradient_accumulation_steps=2, kwargs_handlers=[ddp_kwargs])
    DEVICE = accelerator.device

    if accelerator.is_main_process:
        print(f">> 设备: {DEVICE}")

    l2_centers = torch.load(task_cfg["l2_centers"],  map_location="cpu", weights_only=False).to(torch.bfloat16)
    l2_widths  = torch.load(task_cfg["l2_widths"],   map_location="cpu", weights_only=False).to(torch.bfloat16)
    l1_centers = torch.load(task_cfg["l1_centers"],  map_location="cpu", weights_only=False).to(torch.bfloat16)
    l1_widths  = torch.load(task_cfg["l1_widths"],   map_location="cpu", weights_only=False).to(torch.bfloat16)
    m_global   = torch.load(task_cfg["m_global"],    map_location="cpu", weights_only=False).to(torch.bfloat16)

    num_boxes = l2_centers.shape[0]
    dim = l2_centers.shape[1]
    num_tools = task_cfg["old_num_tools"]

    router = DualStreamRouter(
        dim=dim, num_tools=num_tools, num_boxes=num_boxes, m_global=m_global,
        l2_centers=l2_centers, l2_widths=l2_widths,
        l1_centers=l1_centers, l1_widths=l1_widths
    ).to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    base_llm = AutoModelForCausalLM.from_pretrained(LLAMA_PATH, torch_dtype=torch.bfloat16).to(DEVICE)
    base_llm.eval()
    for param in base_llm.parameters():
        param.requires_grad = False

    llm_caller = LLMCaller(
        base_llm=base_llm, tokenizer=tokenizer,
        num_boxes=num_boxes, num_tools=num_tools
    ).to(DEVICE)

    # 加载离线 Router 权重（如果存在）
    off_ckpt = task_cfg.get("offline_router_path")
    if off_ckpt and os.path.exists(off_ckpt):
        if accelerator.is_main_process:
            print(f">> 注入离线 Router: {off_ckpt}")
        off_router = torch.load(off_ckpt, map_location="cpu")
        router.load_state_dict(off_router['router_state_dict'], strict=False)
        llm_caller.query_proj.load_state_dict(off_router['query_proj_state_dict'])

    router.eval()
    for param in router.parameters():
        param.requires_grad = False

    llm_caller.query_proj.train()
    for param in llm_caller.query_proj.parameters():
        param.requires_grad = False  # 冻结

    loss_fn = IHLoss()
    dataloader = get_base_dataloader(batch_size=task_cfg["batch_size"])

    trainer = BaseTrainer(
        router=router, llm_caller=llm_caller,
        loss_fn=loss_fn, dataloader=dataloader,
        device=DEVICE, accelerator=accelerator
    )

    os.makedirs(task_cfg["ckpt_dir"], exist_ok=True)

    history_total, history_task, history_geo, history_cont = [], [], [], []
    best_loss = float('inf')
    patience_counter = 0
    patience = 5  # Early Stop 耐心值

    for epoch in range(task_cfg["epochs"]):
        if accelerator.is_main_process:
            print(f"\n--- Epoch {epoch} ---")
        t_loss, task_loss, geo_loss, cont_loss = trainer.train_epoch(epoch)

        if accelerator.is_main_process:
            history_total.append(t_loss)
            history_task.append(task_loss)
            history_geo.append(geo_loss)
            history_cont.append(cont_loss)

            if t_loss < best_loss and t_loss > 0:
                best_loss = t_loss
                patience_counter = 0
                trainer.save_checkpoint(epoch, save_dir=task_cfg["ckpt_dir"])
                print(f"  -> 保存最佳模型（loss={t_loss:.4f}）")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n🛑 早停！连续 {patience} 个 epoch 未改善，退出训练。")
                    break

        accelerator.wait_for_everyone()

    if accelerator.is_main_process and history_total:
        history = {
            "total": history_total,
            "task":  history_task,
            "geo":   history_geo,
            "cont":  history_cont,
        }
        save_loss_history(task_cfg, history)

        task_name_lower = task_cfg["name"].lower().replace(" ", "_")
        png_filename = os.path.join(task_cfg["ckpt_dir"], f"{task_name_lower}_loss_curve.png")
        print(f"\n📊 训练完毕，正在生成 Loss 曲线图...")
        plot_loss_curve(history, f'{task_cfg["name"]} Training Loss Curve', png_filename)

    if accelerator.is_main_process:
        print(f"\n✅ {task_cfg['name']} 训练完成！权重保存在: {task_cfg['ckpt_dir']}")


# ============================================================================
# Mode: train (CL Task1/2/3)
# ============================================================================
def _import_cl_modules():
    from models.router import DualStreamRouter
    from models.llm_caller_cl import LLMCaller as CLCaller
    from training.trainer_cl import IH_Trainer as CLTrainer
    from training.losses import IHLoss
    from data_process.dataset_factory import build_dataloader
    return DualStreamRouter, CLCaller, CLTrainer, IHLoss, build_dataloader


def run_train_cl(task_cfg, prev_task_cfg, args):
    print(f"\n{'='*50}")
    print(f"[Train] {task_cfg['name']} — 持续学习训练（CL 版）")
    print(f"{'='*50}")

    DualStreamRouter, CLCaller, CLTrainer, IHLoss, build_dataloader = _import_cl_modules()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(gradient_accumulation_steps=2, kwargs_handlers=[ddp_kwargs])
    DEVICE = accelerator.device

    if accelerator.is_main_process:
        print(f">> 设备: {DEVICE}")

    # --- 加载物理空间（当前任务的扩容版本） ---
    l2_centers = torch.load(task_cfg["l2_centers"],  map_location="cpu", weights_only=False).to(torch.bfloat16)
    l2_widths  = torch.load(task_cfg["l2_widths"],   map_location="cpu", weights_only=False).to(torch.bfloat16)
    l1_centers = torch.load(task_cfg["l1_centers"],  map_location="cpu", weights_only=False).to(torch.bfloat16)
    l1_widths  = torch.load(task_cfg["l1_widths"],   map_location="cpu", weights_only=False).to(torch.bfloat16)
    m_global   = torch.load(task_cfg["m_global"],     map_location="cpu", weights_only=False).to(torch.bfloat16)

    total_num_boxes = l2_centers.shape[0]
    total_num_tools = m_global.shape[0]
    dim = l2_centers.shape[1]

    if accelerator.is_main_process:
        print(f">> 物理空间: {total_num_boxes} 盒子, {total_num_tools} 工具")

    # --- 初始化 Router ---
    router = DualStreamRouter(
        dim=dim, num_tools=total_num_tools, num_boxes=total_num_boxes, m_global=m_global,
        l2_centers=l2_centers, l2_widths=l2_widths,
        l1_centers=l1_centers, l1_widths=l1_widths
    ).to(DEVICE)

    # 半冻结 Router（只解冻新盒子坐标）
    router.eval()
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

    # --- 注入旧任务记忆（必须先加载，才能正确初始化 llm_caller） ---
    old_ckpt_path = args.old_checkpoint if args.old_checkpoint else task_cfg.get("old_checkpoint")
    old_ckpt = None
    if old_ckpt_path and os.path.exists(old_ckpt_path):
        if accelerator.is_main_process:
            print(f">> 注入旧任务记忆: {old_ckpt_path}")
        old_ckpt = torch.load(old_ckpt_path, map_location="cpu")
        # 从 checkpoint 的 tensor shape 反推真实 old 值（不依赖 config 字典）
        real_old_num_boxes = old_ckpt['prompt_pool'].shape[0]
        cls_state = old_ckpt['classifier_state_dict']
        if 'weight_old' in cls_state:
            real_old_num_tools = cls_state['weight_old'].shape[0]
        else:
            real_old_num_tools = cls_state['weight'].shape[0]
        if accelerator.is_main_process:
            print(f">> checkpoint 物理空间: {real_old_num_boxes} 盒子, {real_old_num_tools} 工具")
    else:
        real_old_num_boxes = prev_task_cfg["old_num_boxes"]
        real_old_num_tools = prev_task_cfg["old_num_tools"]
        if accelerator.is_main_process:
            print(f"⚠️ 找不到旧权重: {old_ckpt_path}，从 config 字典 fallback（582/11112）")

    # --- 初始化 CL 版 LLM Caller（用真实 old 值） ---
    llm_caller = CLCaller(
        base_llm=base_llm, tokenizer=tokenizer,
        num_boxes=total_num_boxes, num_tools=total_num_tools,
        old_num_boxes=real_old_num_boxes, old_num_tools=real_old_num_tools
    ).to(DEVICE)

    if old_ckpt is not None:
        old_router_state = old_ckpt['router_state_dict']
        with torch.no_grad():
            router.l2_centers.data[:real_old_num_boxes] = old_router_state['l2_centers'][:real_old_num_boxes].to(DEVICE)

        # 恢复特征降维层（冻结）
        llm_caller.query_proj.load_state_dict(old_ckpt['query_proj_state_dict'])
        llm_caller.query_proj.eval()
        for param in llm_caller.query_proj.parameters():
            param.requires_grad = False

        # 恢复 Prompt Pool 和 Classifier
        llm_caller.load_base_weights(
            old_prompt_tensor=old_ckpt['prompt_pool'],
            old_classifier_state_dict=old_ckpt['classifier_state_dict']
        )

    # --- 构建 DataLoader（通过 factory） ---
    # 先把数据集专用参数从 task_cfg 中摘出来，避免与 factory 的显式参数冲突
    dataloader = build_dataloader(
        task=task_cfg["name"].lower(),
        batch_size=task_cfg["batch_size"],
        train_json=task_cfg.get("train_json"),
        new_tools_json=task_cfg.get("new_tools_json"),
        tool_to_l2_path=task_cfg.get("tool_to_l2"),
        l2_to_l1_path=task_cfg.get("l2_to_l1"),
        prev_task_tools_json=task_cfg.get("prev_task_tools_json"),
        prev_task_train_json=task_cfg.get("prev_task_train_json"),
    )

    loss_fn = IHLoss()

    trainer = CLTrainer(
        router=router, llm_caller=llm_caller,
        loss_fn=loss_fn, dataloader=dataloader,
        device=DEVICE, accelerator=accelerator
    )

    os.makedirs(task_cfg["ckpt_dir"], exist_ok=True)

    # 初始化 loss 历史记录
    history_total, history_task, history_geo, history_cont = [], [], [], []
    actual_epochs = 0

    for epoch in range(task_cfg["epochs"]):
        if accelerator.is_main_process:
            print(f"\n--- Epoch {epoch} ---")
        t_loss, task_loss, geo_loss, cont_loss, early_stop, is_best = trainer.train_epoch(epoch)

        # 只有主进程记录 loss（其他进程的 loss 可能是无效的）
        if accelerator.is_main_process:
            history_total.append(t_loss)
            history_task.append(task_loss)
            history_geo.append(geo_loss)
            history_cont.append(cont_loss)

        stop_tensor = torch.tensor(1 if early_stop else 0, device=DEVICE)
        stop_tensor = accelerator.reduce(stop_tensor, reduction="max")

        if accelerator.is_main_process and is_best:
            trainer.save_checkpoint(epoch, save_dir=task_cfg["ckpt_dir"])

        if stop_tensor.item() > 0:
            if accelerator.is_main_process:
                print(f"🛑 早停，所有 GPU 退出训练循环！")
            break

        accelerator.wait_for_everyone()
        actual_epochs += 1

    # 绘制 loss 曲线（仅主进程）
    if accelerator.is_main_process and history_total:
        history = {
            "total": history_total,
            "task":  history_task,
            "geo":   history_geo,
            "cont":  history_cont,
        }
        save_loss_history(task_cfg, history)

        task_name_lower = task_cfg["name"].lower().replace(" ", "_")
        png_filename = os.path.join(task_cfg["ckpt_dir"], f"{task_name_lower}_loss_curve.png")
        print(f"\n📊 训练完毕，正在生成 Loss 曲线图...")
        plot_loss_curve(history, f'{task_cfg["name"]} Continual Learning Loss Curve', png_filename)

    if accelerator.is_main_process:
        print(f"\n✅ {task_cfg['name']} 训练完成！权重保存在: {task_cfg['ckpt_dir']}")


# ============================================================================
# Mode: eval
# ============================================================================
def run_eval(model_task_cfg, eval_task_cfg, limit):
    """
    用 model_task_cfg 的 checkpoint，在 eval_task_cfg 的测试集上评估。
    model_task_cfg 和 eval_task_cfg 可以相同（自测），也可以不同（跨任务评估）。
    """
    import datetime

    print(f"\n{'='*50}")
    print(f"[Eval] 用 {model_task_cfg['name']} 的权重，在 {eval_task_cfg['name']} 的测试集上评估")
    print(f"{'='*50}")

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- 加载物理空间（始终用 model_task_cfg 的空间，因为权重来自它） ---
    l2_centers = torch.load(model_task_cfg["l2_centers"],  map_location="cpu", weights_only=False).to(torch.bfloat16)
    l2_widths  = torch.load(model_task_cfg["l2_widths"],   map_location="cpu", weights_only=False).to(torch.bfloat16)
    l1_centers = torch.load(model_task_cfg["l1_centers"],  map_location="cpu", weights_only=False).to(torch.bfloat16)
    l1_widths  = torch.load(model_task_cfg["l1_widths"],   map_location="cpu", weights_only=False).to(torch.bfloat16)
    m_global   = torch.load(model_task_cfg["m_global"],    map_location="cpu", weights_only=False).to(torch.bfloat16)
    tool_to_l2 = torch.load(model_task_cfg["tool_to_l2"],  map_location="cpu", weights_only=False)

    total_num_boxes = l2_centers.shape[0]
    total_num_tools = m_global.shape[0]
    dim = l2_centers.shape[1]

    # --- 加载模型 ---
    from models.router import DualStreamRouter
    from models.llm_caller import LLMCaller

    router = DualStreamRouter(
        dim=dim, num_tools=total_num_tools, num_boxes=total_num_boxes, m_global=m_global,
        l2_centers=l2_centers, l2_widths=l2_widths,
        l1_centers=l1_centers, l1_widths=l1_widths
    ).to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    base_llm = AutoModelForCausalLM.from_pretrained(LLAMA_PATH, torch_dtype=torch.bfloat16).to(DEVICE)

    llm_caller = LLMCaller(
        base_llm=base_llm, tokenizer=tokenizer,
        num_boxes=total_num_boxes, num_tools=total_num_tools
    ).to(DEVICE)

    # --- 加载最新 checkpoint ---
    ckpt_dir = model_task_cfg["ckpt_dir"]
    ckpt_files = []
    if os.path.exists(ckpt_dir):
        ckpt_files = sorted(
            [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")],
            key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x))
        )

    if not ckpt_files:
        print(f"❌ 未找到 checkpoint 文件: {ckpt_dir}")
        return

    ckpt_path = os.path.join(ckpt_dir, ckpt_files[-1])
    mtime = os.path.getmtime(ckpt_path)
    mtime_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    print(f">> 加载 checkpoint: {ckpt_path}")
    print(f">> 文件修改时间: {mtime_str}  (epoch {len(ckpt_files)})")
    print(f">> 提示: 确保这是你刚才训练结束保存的 epoch（早停在 epoch {len(ckpt_files)}）")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    router.load_state_dict(checkpoint['router_state_dict'])

    llm_caller_state = {}
    for k, v in checkpoint.get('query_proj_state_dict', {}).items():
        llm_caller_state[f'query_proj.{k}'] = v
    for k, v in checkpoint.get('classifier_state_dict', {}).items():
        llm_caller_state[f'classifier.{k}'] = v
    if 'prompt_pool' in checkpoint:
        llm_caller_state['prompt_pool'] = checkpoint['prompt_pool'].to(DEVICE)

    llm_caller.load_state_dict(llm_caller_state, strict=False)

    router.eval()
    llm_caller.eval()

    # --- 加载工具映射（用被测任务的映射，权重来自 model_task_cfg） ---
    name_to_id = load_data_mappings(eval_task_cfg)

    # --- 解析测试集（用被测任务的 eval_json） ---
    eval_json = eval_task_cfg["eval_json"]
    if not os.path.exists(eval_json):
        print(f"❌ 测试文件不存在: {eval_json}")
        return

    with open(eval_json, "r", encoding="utf-8") as f:
        data = json.load(f)

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
        if target_tool_id not in tool_to_l2:
            continue

        valid_samples += 1

        with torch.no_grad():
            q_vec, inputs_embeds, attention_mask = llm_caller.extract_query_vector([query_text])
            t_prev = torch.tensor([-1], dtype=torch.long, device=DEVICE)
            s_total, _ = router(q_vec, t_prev)
            pred_l2_id = s_total.argmax(dim=-1)
            logits = llm_caller(inputs_embeds, attention_mask, pred_l2_id)

            _, top_k_tool_ids = torch.topk(logits, k=5, dim=-1)
            top_k_tools_list = top_k_tool_ids[0].tolist()

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

    # --- 打印报告 ---
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


# ============================================================================
# 主入口
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="IH-PromptDSI 统一入口")
    parser.add_argument(
        "--task", default="all",
        choices=["base", "task1", "task2", "task3", "all"],
        help="目标任务（默认 all）"
    )
    parser.add_argument(
        "--mode", default="all",
        choices=["prepare", "train", "eval", "all"],
        help="运行模式（默认 all）"
    )
    parser.add_argument(
        "--stage", type=int, default=3,
        choices=[0, 1, 2, 3],
        help="prepare 阶段：0=检查，1=提取工具，2=归一化，3=聚类（默认 3）"
    )
    parser.add_argument(
        "--limit", type=int, default=-1,
        help="评估样本上限（-1=全部，默认 -1）"
    )
    parser.add_argument(
        "--eval-tasks", type=str, default=None,
        help="评估时指定要测试的任务列表，逗号分隔（覆盖 --task）。"
             "例如 --eval-tasks task1,train 用 task2 的 checkpoint 跑 task1 和 train 的测试集"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=None,
        help="GPU 数量（默认 accelerate 自动检测）"
    )
    parser.add_argument(
    "--old-checkpoint", type=str, default=None,
    help="CL 训练时注入旧任务记忆的 checkpoint 路径（默认自动找最新的）"
    )
    args = parser.parse_args()

    # 确定要运行的任务列表
    if args.task == "all":
        task_names = ["base", "task1", "task2", "task3"]
    else:
        task_names = [args.task]

    for task_name in task_names:
        if task_name not in TASK_CONFIGS:
            print(f"❌ 未知任务: {task_name}")
            continue

        task_cfg = TASK_CONFIGS[task_name]
        prev_task_cfg = TASK_CONFIGS[task_cfg["prev_task"]] if task_cfg["prev_task"] else None

        print(f"\n\n{'#'*60}")
        print(f"## 处理任务: {task_cfg['name']}")
        print(f"{'#'*60}")

        # --- Mode: prepare ---
        if args.mode in ("prepare", "all"):
            # stage 1+2 只需跑一次（stage 1/2 有幂等保护）
            if args.stage <= 1:
                run_prepare(task_name, stage=1, num_gpus=args.num_gpus)
            if args.stage <= 2:
                run_prepare(task_name, stage=2, num_gpus=args.num_gpus)
            if args.stage >= 3:
                run_prepare(task_name, stage=3, num_gpus=args.num_gpus)

        # --- Mode: train ---
        if args.mode in ("train", "all"):
            if task_name == "base":
                run_train_base(task_cfg, args)
            else:
                run_train_cl(task_cfg, prev_task_cfg, args)

        # --- Mode: eval ---
        if args.mode in ("eval", "all"):
            eval_limit = args.limit if args.limit > 0 else -1  # -1 = 全量
            # 确定要评估的任务列表（可能和训练 --task 不同）
            if args.mode == "eval" and args.eval_tasks:
                eval_task_names = [t.strip() for t in args.eval_tasks.split(",")]
                # 支持别名映射
                alias_map = {"train": "base", "base": "base",
                             "task1": "task1", "task2": "task2", "task3": "task3"}
                eval_task_names = [alias_map.get(t, t) for t in eval_task_names]
                eval_task_names = [t for t in eval_task_names if t in TASK_CONFIGS]
                if not eval_task_names:
                    print(f"⚠️  --eval-tasks 中没有合法任务（train/base/task1/task2/task3）")
                    return
            else:
                eval_task_names = [task_name]

            for et in eval_task_names:
                if et not in TASK_CONFIGS:
                    print(f"⚠️  未知任务 {et}，跳过")
                    continue
                run_eval(task_cfg, eval_task_cfg=TASK_CONFIGS[et], limit=eval_limit)

    print(f"\n\n{'='*50}")
    print(f"✅ 全部完成！")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
