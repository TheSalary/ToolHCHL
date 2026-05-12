from torch.utils.data import DataLoader


def build_dataloader(task="base", batch_size=4, shuffle=True, num_workers=4,
                     train_json=None, new_tools_json=None,
                     tool_to_l2_path=None, l2_to_l1_path=None,
                     prev_task_tools_json=None, prev_task_train_json=None,
                     replay_per_tool=1):
    if task == "base":
        from data_process.dataset import IH_ToolDataset as BaseDataset
        dataset = BaseDataset()
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    elif task in ("task1", "task2", "task3"):
        from data_process.dataset_cl import IH_ToolDataset_CL as CLDataset

        # Base 阶段的老工具和老数据（固定路径）
        base_tools_json = "./data/train/raw/train_tools_with_id.json"
        base_train_json = "./data/train/raw/retrieval_train.json"

        # 历史任务回放路径，支持 str 或 None，统一转为 list
        prev_tools_list = [prev_task_tools_json] if isinstance(prev_task_tools_json, str) else (prev_task_tools_json or [])
        prev_json_list  = [prev_task_train_json] if isinstance(prev_task_train_json, str)  else (prev_task_train_json  or [])

        dataset = CLDataset(
            task=task,
            task_tools_json=new_tools_json,
            task_train_json=train_json,
            tool_to_l2_path=tool_to_l2_path,
            l2_to_l1_path=l2_to_l1_path,
            replay_per_tool=replay_per_tool,
            replay_task_tools=prev_tools_list,
            replay_task_jsons=prev_json_list,
            replay_base_tools_json=base_tools_json,
            replay_base_train_json=base_train_json,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    else:
        raise ValueError(f"Unknown task: {task}")
