import torch
import json
import os
import re
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def normalize_name(name):
    """暴力标准化，用于绝对匹配"""
    return re.sub(r'[^a-zA-Z0-9]', '', str(name)).lower()


class IH_ToolDataset_CL(Dataset):
    """
    通用 CL 数据集，根据传入的任务配置动态加载当前任务数据 + 所有历史任务回放。
    replay_tasks: list of dicts, 每个 dict 包含 replay 所需的路径
    """
    def __init__(self,
                 task,                      # "task1" / "task2" / "task3"
                 task_tools_json,           # 当前任务的新工具字典路径
                 task_train_json,            # 当前任务的新训练数据路径
                 tool_to_l2_path,           # 当前任务的 tool->L2 映射
                 l2_to_l1_path,             # 当前任务的 L2->L1 映射
                 replay_per_tool=1,
                 replay_task_tools=None,    # list of 历史任务的新工具字典路径（不含 base）
                 replay_task_jsons=None,     # list of 历史任务的训练数据路径
                 replay_base_tools_json=None,
                 replay_base_train_json=None):
        """
        replay_task_tools / replay_task_jsons:  按顺序传入每个历史任务的新工具+新数据路径
        replay_base_tools_json / replay_base_train_json: Base 阶段的老工具字典和训练数据
        """
        self.device = "cpu"
        self.name_to_id = {}

        print(f"1. 正在加载当前任务 [{task}] 的工具字典与映射表...")

        # --- 加载当前任务（TaskN）的新工具 ---
        self._load_tools_from_json(task_tools_json, label="新工具")

        # --- 加载历史任务回放工具 ---
        replay_sources = []
        if replay_base_tools_json and replay_base_train_json:
            replay_sources.append(("Base", replay_base_tools_json, replay_base_train_json))
        if replay_task_tools and replay_task_jsons:
            for pt, pj in zip(replay_task_tools, replay_task_jsons):
                label = os.path.basename(os.path.dirname(os.path.dirname(pt)))  # e.g. "task1"
                replay_sources.append((label, pt, pj))

        for label, tools_json, train_json in replay_sources:
            count_before = len(self.name_to_id)
            self._load_tools_from_json(tools_json, label=f"{label}回放")
            added = len(self.name_to_id) - count_before
            replay_sources_dict = getattr(self, '_replay_sources', [])
            replay_sources_dict.append((label, tools_json, train_json, added))
            self._replay_sources = replay_sources_dict

        print(f"   -> 🔗 内存大字典构建完毕！总容量: {len(self.name_to_id)}")

        # 加载物理映射表（当前任务的，覆盖所有工具）
        self.tool_to_l2 = torch.load(tool_to_l2_path, weights_only=False)
        self.l2_to_l1 = torch.load(l2_to_l1_path, weights_only=False)

        print(f"2. 正在解析训练数据并构建混合回放...")

        # 当前任务新数据
        with open(task_train_json, "r", encoding="utf-8") as f:
            task_raw = json.load(f)
        new_samples = self._parse_conversations(task_raw, is_new_task=True, task_label=task)
        print(f"   -> 成功提取 [{task}] 新数据: {len(new_samples)} 条")

        # 历史任务回放（每工具随机抽 replay_per_tool 条）
        all_replay_samples = []
        if replay_base_tools_json and replay_base_train_json:
            with open(replay_base_train_json, "r", encoding="utf-8") as f:
                base_raw = json.load(f)
            bs = self._parse_conversations(base_raw, is_new_task=False, task_label="Base")
            all_replay_samples.extend(self._replay_grouped(bs, replay_per_tool))
            print(f"   -> 成功抽取 [Base] 回放数据: {len(all_replay_samples)} 条")

        if replay_task_tools and replay_task_jsons:
            for pt, pj in zip(replay_task_tools, replay_task_jsons):
                label = os.path.basename(os.path.dirname(os.path.dirname(pt)))
                with open(pj, "r", encoding="utf-8") as f:
                    prev_raw = json.load(f)
                ps = self._parse_conversations(prev_raw, is_new_task=False, task_label=label)
                count = len(all_replay_samples)
                all_replay_samples.extend(self._replay_grouped(ps, replay_per_tool))
                print(f"   -> 成功抽取 [{label}] 回放数据: {len(all_replay_samples) - count} 条")

        # 混合 + 打乱
        final_samples = new_samples + all_replay_samples
        random.shuffle(final_samples)
        self.samples = final_samples
        print(f"✅ 数据集加载完毕！共 {len(self.samples)} 条混合训练数据。")

    def _load_tools_from_json(self, tools_json, label="工具"):
        """从 dict 或 list 格式的工具字典中提取 name_to_id 映射"""
        with open(tools_json, "r", encoding="utf-8") as f:
            tools = json.load(f)

        count = 0
        if isinstance(tools, dict):
            for t_id_str, text in tools.items():
                if isinstance(text, str):
                    t_name, a_name = "", ""
                    a_match = re.search(r'API:\s*(.*?)\.(?:\s*API Description:|$)', text)
                    if a_match:
                        a_name = a_match.group(1).strip()
                    t_match = re.search(r'Tool:\s*(.*?)\.(?:\s*Description:|\s*API:)', text)
                    if t_match:
                        t_name = t_match.group(1).strip()
                    if t_name and a_name:
                        norm_str = normalize_name(f"{a_name}for{t_name}")
                        self.name_to_id[norm_str] = int(t_id_str)
                        count += 1
                elif isinstance(text, dict):
                    t_name, a_name = "", ""
                    t_match = re.search(r'Tool:\s*(.*?)\.(?:\s*Description:|\s*API:)', text.get("text", ""))
                    a_match = re.search(r'API:\s*(.*?)\.(?:\s*API Description:|$)', text.get("text", ""))
                    if t_match:
                        t_name = t_match.group(1).strip()
                    if a_match:
                        a_name = a_match.group(1).strip()
                    if t_name and a_name:
                        norm_str = normalize_name(f"{a_name}for{t_name}")
                        self.name_to_id[norm_str] = int(t_id_str)
                        count += 1
        elif isinstance(tools, list):
            for t in tools:
                api_n = normalize_name(t.get("api_name", ""))
                tool_n = normalize_name(t.get("tool_name", ""))
                self.name_to_id[f"{api_n}for{tool_n}"] = t.get("tool_id", t.get("id"))
                count += 1
        print(f"   -> 成功加载 {label}: {count} 个")

    def _parse_conversations(self, data_list, is_new_task=False, task_label=""):
        """解析 conversations 格式的数据，返回样本列表"""
        parsed = []
        for item in tqdm(data_list, desc=f"解析 {task_label} JSON"):
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
            tool_name = match.group(1).strip()
            api_name = match.group(2).strip()
            target_norm = normalize_name(f"{api_name}for{tool_name}")
            if target_norm not in self.name_to_id:
                continue
            target_id = self.name_to_id[target_norm]
            if target_id not in self.tool_to_l2:
                continue
            parsed.append({
                "query_text": query_text,
                "t_prev_id": -1,
                "target_tool_id": target_id,
                "target_str": target_str,
                "source_task": task_label,
            })
        return parsed

    def _replay_grouped(self, samples, replay_per_tool):
        """按 tool_id 分组，每组随机抽 replay_per_tool 条"""
        grouped = {}
        for s in samples:
            tid = s["target_tool_id"]
            grouped.setdefault(tid, []).append(s)
        replay = []
        for tid, items in grouped.items():
            sampled = random.sample(items, min(replay_per_tool, len(items)))
            replay.extend(sampled)
        return replay

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        target_tool_id = sample["target_tool_id"]
        target_l2 = self.tool_to_l2[target_tool_id]
        target_l1 = self.l2_to_l1.get(target_l2, 0)
        return {
            "t_prev_id": torch.tensor(sample["t_prev_id"], dtype=torch.long),
            "target_l2_id": torch.tensor(target_l2, dtype=torch.long),
            "target_l1_id": torch.tensor(target_l1, dtype=torch.long),
            "tool_label": torch.tensor(target_tool_id, dtype=torch.long),
            "query_text": sample["query_text"],
            "target_generation_text": sample["target_str"],
        }
