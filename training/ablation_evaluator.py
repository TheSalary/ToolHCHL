"""
training/ablation_evaluator.py
=============================
消融实验统一评估器。

从 run_ablation.py 的 run_eval / run_ablation_compare 提取并重构，
依赖 models/router.py 和 models/llm_caller.py 的新工厂体系。

主要功能
--------
- 统一推理评估：支持所有 Router 类型和消融模式
- 自动 Router 加载和状态恢复（兼容 CL / 基线 checkpoint 格式）
- 推理时消融：AblationRouter 包装器（无需重新训练）
- Recall@K / NDCG@K 指标计算

使用示例
--------
    from training.ablation_evaluator import AblationEvaluator, run_ablation_compare

    evaluator = AblationEvaluator(device="cuda:0")
    metrics = evaluator.eval(
        model_cfg=task_cfg,
        eval_cfg=task_cfg,
        ablation="router_semantic",
        eval_json="path/to/eval.json",
        limit=50,
    )
"""

import os
import re
import json
import math
import datetime
import torch
import torch.nn as nn
from tqdm import tqdm

from config import normalize, load_data_mappings, get_checkpoint_dir


# ============================================================================
# 1. AblationEvaluator — 统一评估器
# ============================================================================
class AblationEvaluator:
    """
    统一的推理评估器。

    支持：
    - 所有 Router 类型（DualStreamRouter / FlatDualStreamRouter / SimpleLinearRouter）
    - 所有推理时消融模式（router_semantic / router_dependency / router_no_gate）
    - CL 和基线两种 checkpoint 格式
    """

    def __init__(self, device=None):
        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )

    def _load_checkpoint(self, ckpt_path):
        """加载 checkpoint 并解析格式（CL / 基线）。"""
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        router_state = ckpt['router_state_dict']
        cls_state = ckpt.get('classifier_state_dict', {})

        has_l2_centers = 'l2_centers' in router_state

        if has_l2_centers:
            ckpt_l2 = router_state['l2_centers']
            ckpt_num_boxes = ckpt_l2.shape[0]
            dim = ckpt_l2.shape[1]
        else:
            prompt_proj = router_state.get('prompt_proj.weight')
            dim = prompt_proj.shape[1] if prompt_proj is not None else 4096
            ckpt_num_boxes = ckpt['prompt_pool'].shape[0]

        if 'weight_old' in cls_state:
            ckpt_old_tools = cls_state['weight_old'].shape[0]
            ckpt_new_tools = cls_state['weight_new'].shape[0]
            ckpt_num_tools = ckpt_old_tools + ckpt_new_tools
            is_cl_format = True
            ckpt_old_boxes = ckpt.get('old_num_boxes', ckpt['prompt_pool'].shape[0])
        else:
            ckpt_num_tools = cls_state['weight'].shape[0]
            is_cl_format = False
            ckpt_old_boxes = None

        return {
            'router_state': router_state,
            'cls_state': cls_state,
            'ckpt': ckpt,
            'has_l2_centers': has_l2_centers,
            'ckpt_num_boxes': ckpt_num_boxes,
            'ckpt_num_tools': ckpt_num_tools,
            'ckpt_old_tools': cls_state.get('weight_old', torch.tensor(0)).shape[0] if is_cl_format else None,
            'ckpt_new_tools': cls_state.get('weight_new', torch.tensor(0)).shape[0] if is_cl_format else None,
            'is_cl_format': is_cl_format,
            'ckpt_old_boxes': ckpt_old_boxes,
            'dim': dim,
        }

    def _build_router(self, model_cfg, ablation_info, ablation_name):
        """根据消融模式构建 Router（可能包装 AblationRouter）。"""
        from models.router import (
            DualStreamRouter, FlatDualStreamRouter, SimpleLinearRouter,
            create_router, ABLATION_TO_ROUTER_TYPE,
        )
        from training.ablation_components import AblationRouter

        info = ablation_info
        dim = info['dim']
        num_tools = info['ckpt_num_tools']
        num_boxes = info['ckpt_num_boxes']

        if info['has_l2_centers']:
            l2_c = info['router_state']['l2_centers'].detach().clone().to(torch.bfloat16)
            l2_w = info['router_state']['l2_widths'].detach().clone().to(torch.bfloat16)
        else:
            l2_c = l2_w = None

        m_global_use = None
        if info['has_l2_centers']:
            m_gl = torch.load(model_cfg["m_global"], map_location=self.device,
                              weights_only=False).to(torch.bfloat16)
            m_global_use = m_gl[:num_tools, :num_boxes]

        l1_c = l1_w = None
        if info['has_l2_centers']:
            l1_c_t = torch.load(model_cfg["l1_centers"], map_location=self.device,
                                weights_only=False).to(torch.bfloat16)
            l1_w_t = torch.load(model_cfg["l1_widths"], map_location=self.device,
                               weights_only=False).to(torch.bfloat16)
            l1_c = l1_c_t[:l2_c.shape[0] // 10]  # 近似
            l1_w = l1_w_t[:l2_c.shape[0] // 10]

        is_inference_ab = ablation_name in {
            "router_semantic", "router_dependency", "router_no_gate"
        }

        if is_inference_ab:
            base_router = DualStreamRouter(
                dim=dim, num_tools=num_tools, num_boxes=num_boxes,
                m_global=m_global_use,
                l2_centers=l2_c, l2_widths=l2_w,
                l1_centers=l1_c, l1_widths=l1_w,
            ).to(self.device)
            router = AblationRouter(base_router, ablation_mode=ablation_name)
        elif ablation_name == "linear_router":
            router = SimpleLinearRouter(
                dim=dim, num_tools=num_tools, num_boxes=num_boxes,
            ).to(self.device)
        elif ablation_name in ("w/o_hierarchy", "flat_space"):
            m_tool = m_global_use
            router = FlatDualStreamRouter(
                dim=dim, num_tools=num_tools, num_boxes=num_boxes,
                m_global_tool=m_tool,
                l2_centers=l2_c, l2_widths=l2_w,
            ).to(self.device)
        else:
            router = DualStreamRouter(
                dim=dim, num_tools=num_tools, num_boxes=num_boxes,
                m_global=m_global_use,
                l2_centers=l2_c, l2_widths=l2_w,
                l1_centers=l1_c, l1_widths=l1_w,
            ).to(self.device)

        return router

    def _build_llm_caller(self, model_cfg, ablation_info, base_llm, tokenizer):
        """构建 LLMCaller（CL / 基线）。"""
        from models.llm_caller import LLMCallerBase, LLMCallerCL

        info = ablation_info
        num_boxes = info['ckpt_num_boxes']
        num_tools = info['ckpt_num_tools']

        if info['is_cl_format']:
            llm_caller = LLMCallerCL(
                base_llm=base_llm, tokenizer=tokenizer,
                num_boxes=num_boxes, num_tools=num_tools,
                old_num_boxes=info['ckpt_old_boxes'],
                old_num_tools=info['ckpt_old_tools'],
            ).to(self.device)
        else:
            llm_caller = LLMCallerBase(
                base_llm=base_llm, tokenizer=tokenizer,
                num_boxes=num_boxes, num_tools=num_tools,
            ).to(self.device)

        return llm_caller

    def _load_llm_caller_state(self, llm_caller, info):
        """将 checkpoint 权重加载到 LLMCaller。"""
        cls_state = info['cls_state']
        ckpt = info['ckpt']

        llm_state = {}
        for k, v in ckpt.get('query_proj_state_dict', {}).items():
            llm_state[f'query_proj.{k}'] = v

        if info['is_cl_format']:
            llm_state['classifier_old.weight'] = cls_state['weight_old'].to(self.device)
            llm_state['classifier_old.bias']   = cls_state['bias_old'].to(self.device)
            llm_state['classifier_new.weight'] = cls_state['weight_new'].to(self.device)
            llm_state['classifier_new.bias']   = cls_state['bias_new'].to(self.device)
            pp = ckpt.get('prompt_pool')
            if pp is not None:
                llm_state['prompt_pool_old'] = pp[:info['ckpt_old_boxes']].to(self.device)
        else:
            for k, v in cls_state.items():
                llm_state[f'classifier.{k}'] = v
            if 'prompt_pool' in ckpt:
                llm_state['prompt_pool'] = ckpt['prompt_pool'].to(self.device)

        llm_caller.load_state_dict(llm_state, strict=False)

    def eval(self, model_cfg, eval_cfg, ablation=None, eval_json=None,
             tool_to_l2=None, name_to_id=None, limit=-1, verbose=True):
        """
        执行推理评估。

        参数:
            model_cfg:     模型配置字典（TASK_CONFIGS 中的条目）
            eval_cfg:     评估数据集配置
            ablation:      消融模式（None / "router_semantic" 等）
            eval_json:     评估数据 JSON 文件路径（优先使用，否则从 eval_cfg 读取）
            tool_to_l2:    工具到 L2 盒子映射（Tensor 或 dict）
            name_to_id:    工具名到 ID 的映射字典
            limit:         评估样本上限（-1=全部）
            verbose:       是否打印调试信息
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from config import LLAMA_PATH
        from models.router import create_router
        from training.ablation_components import AblationRouter

        ablation = ablation or None

        # 加载 checkpoint
        ckpt_dir = get_checkpoint_dir(model_cfg["name"], ablation)
        if not os.path.exists(ckpt_dir):
            print(f"❌ checkpoint 目录不存在: {ckpt_dir}")
            return None

        ckpt_files = sorted(
            [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")],
            key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)),
        )
        if not ckpt_files:
            print(f"❌ 未找到 checkpoint: {ckpt_dir}")
            return None

        ckpt_path = os.path.join(ckpt_dir, ckpt_files[-1])
        mtime_str = datetime.datetime.fromtimestamp(
            os.path.getmtime(ckpt_path)
        ).strftime("%Y-%m-%d %H:%M:%S")
        if verbose:
            print(f">> 加载 checkpoint: {ckpt_path}  ({mtime_str})")

        info = self._load_checkpoint(ckpt_path)

        # 构建 Router
        router = self._build_router(model_cfg, info, ablation)
        router.load_state_dict(info['router_state'])
        router.eval()

        # 加载 LLM
        tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH)
        tokenizer.pad_token = tokenizer.eos_token

        base_llm = AutoModelForCausalLM.from_pretrained(
            LLAMA_PATH, torch_dtype=torch.bfloat16,
        ).to(self.device)

        llm_caller = self._build_llm_caller(model_cfg, info, base_llm, tokenizer)
        self._load_llm_caller_state(llm_caller, info)
        llm_caller.eval()

        # 加载工具映射
        if tool_to_l2 is None:
            t2l = torch.load(model_cfg["tool_to_l2"], map_location="cpu", weights_only=False)
        else:
            t2l = tool_to_l2

        if name_to_id is None:
            name_to_id = load_data_mappings(eval_cfg)

        if isinstance(t2l, dict):
            t2l_map = {int(tid): bid for tid, bid in t2l.items()
                       if int(tid) < info['ckpt_num_tools']}
        else:
            t2l_map = t2l[:info['ckpt_num_tools']]

        # 加载评估数据
        if eval_json is None:
            eval_json = eval_cfg["eval_json"]
        if not os.path.exists(eval_json):
            print(f"❌ 评估文件不存在: {eval_json}")
            return None

        with open(eval_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        data_to_eval = data if limit <= 0 else data[:limit]
        if verbose:
            print(f"\n开始评估（{len(data_to_eval)} 条）...")

        # 推理循环
        valid = 0
        r1, r3, r5 = 0, 0, 0
        n1, n3, n5 = 0.0, 0.0, 0.0

        pbar = tqdm(data_to_eval, desc="评估", unit="条", ncols=80,
                    disable=not verbose)
        for item in pbar:
            convs = item.get("conversations", [])
            query_text, target_str = "", ""
            for conv in convs:
                if conv.get("role") == "user":
                    query_text = conv.get("content", "").strip()
                elif conv.get("role") == "assistant":
                    target_str = conv.get("content", "").strip()

            if not query_text or not target_str:
                continue

            m = re.match(r'<<(.+?)&&(.+?)>>', target_str)
            if not m:
                continue

            target_norm = normalize(f"{m.group(2).strip()}for{m.group(1).strip()}")
            if target_norm not in name_to_id:
                continue

            target_id = name_to_id[target_norm]
            if target_id not in t2l_map:
                continue

            valid += 1

            with torch.no_grad():
                q_vec, input_ids, attention_mask = llm_caller.extract_query_vector([query_text])
                t_prev = torch.tensor([-1], dtype=torch.long, device=self.device)
                s_total, _ = router(q_vec, t_prev)
                pred_l2_id = s_total.argmax(dim=-1)
                logits = llm_caller(input_ids, attention_mask, pred_l2_id)

                num_logits = logits.shape[-1]
                _, top_k = torch.topk(logits, k=min(5, num_logits), dim=-1)
                top_k_list = top_k[0].tolist()

                if target_id >= num_logits:
                    continue

            if target_id in top_k_list[:1]:
                r1 += 1
                n1 += 1.0 / math.log2(1 + 1)
            if target_id in top_k_list[:3]:
                r3 += 1
                n3 += 1.0 / math.log2(top_k_list.index(target_id) + 1 + 1)
            if target_id in top_k_list[:5]:
                r5 += 1
                n5 += 1.0 / math.log2(top_k_list.index(target_id) + 1 + 1)

            if valid > 0:
                pbar.set_postfix_str(f"R@1={r1/valid*100:.1f}%")

        print(f"\n{'='*50}")
        print(f"🎯 评估完成！共测试 {valid} 条数据。")
        if valid > 0:
            print(f"🛠️  Recall@1: {r1/valid*100:.2f}%")
            print(f"🛠️  Recall@3: {r3/valid*100:.2f}%")
            print(f"🛠️  Recall@5: {r5/valid*100:.2f}%")
            print(f"🏆  NDCG@1:   {n1/valid*100:.2f}%")
            print(f"🏆  NDCG@3:   {n3/valid*100:.2f}%")
            print(f"🏆  NDCG@5:   {n5/valid*100:.2f}%")
        print(f"{'='*50}")

        return {
            "recall@1": r1/valid*100 if valid > 0 else 0.0,
            "recall@3": r3/valid*100 if valid > 0 else 0.0,
            "recall@5": r5/valid*100 if valid > 0 else 0.0,
            "ndcg@1":   n1/valid*100 if valid > 0 else 0.0,
            "ndcg@3":   n3/valid*100 if valid > 0 else 0.0,
            "ndcg@5":   n5/valid*100 if valid > 0 else 0.0,
        }


# ============================================================================
# 2. 推理时消融综合对比
# ============================================================================
INFERENCE_ABLATIONS = {"router_semantic", "router_dependency", "router_no_gate"}


def run_ablation_compare(task_name, task_cfg, limit=-1, device=None):
    """
    推理时消融综合对比——用基线 checkpoint，依次评估不同推理时消融模式。

    参数:
        task_name: 任务名称（如 "task1"）
        task_cfg:  TASK_CONFIGS 中对应的配置字典
        limit:     评估样本上限（-1=全部）
        device:    设备（如 "cuda:0"）

    返回:
        results_table: [{模式名: 指标}] 列表
    """
    evaluator = AblationEvaluator(device=device)

    results_table = []

    # 基线
    result_base = evaluator.eval(
        model_cfg=task_cfg, eval_cfg=task_cfg,
        ablation=None, limit=limit,
    )
    if result_base:
        results_table.append({"模式": "基线（完整）", **result_base})

    # 各推理时消融
    for ab in sorted(INFERENCE_ABLATIONS):
        result = evaluator.eval(
            model_cfg=task_cfg, eval_cfg=task_cfg,
            ablation=ab, limit=limit, verbose=False,
        )
        if result:
            results_table.append({"模式": ab, **result})

    # 打印汇总
    print(f"\n{'='*70}")
    print(f"🎯 推理时消融对比结果 | 任务={task_name}")
    print(f"{'='*70}")
    header = f"{'模式':<30} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'N@5':>8}"
    print(header)
    print("-" * 70)
    for r in results_table:
        print(f"{r['模式']:<30} "
              f"{r.get('recall@1', 0.0):>7.2f}% {r.get('recall@3', 0.0):>7.2f}% "
              f"{r.get('recall@5', 0.0):>7.2f}% {r.get('ndcg@5', 0.0):>7.2f}%")

    result_file = (f"./ablation_compare_results_{task_name}_"
                   f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results_table, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 结果已保存到: {result_file}")

    return results_table
