import json
import torch
from sentence_transformers import SentenceTransformer
import hdbscan
import numpy as np

class ToolBenchDataProcessor:
    def __init__(self, encoder_model_name="all-MiniLM-L6-v2"):
        # 用于将重写后的意图文本映射为 d 维向量的轻量级 Encoder [cite: 7]
        self.encoder = SentenceTransformer(encoder_model_name)
        
    def llm_intent_rewrite(self, raw_api_doc):
        """
        步骤 1: 意图提取。
        在实际工程中，这里会调用 gpt-3.5 或本地 Llama 模型。
        将长文档重写为标准的: [领域标签] + 动作 + 核心业务对象 格式 。
        """
        # 伪代码：模拟 LLM 的输出
        # 假设原始文档是一段关于汇率转换的复杂描述
        domain = raw_api_doc.get("category_name", "Unknown")
        action = "Convert"
        business_object = "currency exchange rate"
        return f"[{domain}] {action} {business_object}"

    def process_and_cluster(self, raw_toolbench_data):
        """
        核心处理逻辑：重写 -> 分组 -> 局部聚类
        """
        # 用于存储最终处理结果的列表
        processed_data = []
        
        # 字典结构：L1 Domain -> List of Tool Dictionaries
        l1_groups = {}
        
        print("1. 开始意图重写与物理隔离...")
        for tool in raw_toolbench_data:
            # 提取真实领域作为 L1 Box 
            l1_domain = tool["category_name"]
            
            # 意图重写 
            rewritten_intent = self.llm_intent_rewrite(tool)
            
            # 向量化
            intent_vector = self.encoder.encode(rewritten_intent)
            
            tool_entry = {
                "tool_id": tool["tool_id"],
                "name": tool["tool_name"],
                "l1_domain": l1_domain,
                "rewritten_intent": rewritten_intent,
                "vector": intent_vector
            }
            
            if l1_domain not in l1_groups:
                l1_groups[l1_domain] = []
            l1_groups[l1_domain].append(tool_entry)

        print("2. 在 L1 内部执行 HDBSCAN L2 子空间发现...")
        global_l2_counter = 0
        
        # 仅在同一个 L1 内部应用聚类 
        for l1_domain, tools_in_l1 in l1_groups.items():
            vectors_in_l1 = np.array([t["vector"] for t in tools_in_l1])
            
            # 使用 HDBSCAN 发现不同密度的功能簇 
            # min_cluster_size 可以根据具体领域的工具数量动态调整
            clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
            cluster_labels = clusterer.fit_predict(vectors_in_l1)
            
            # 将聚类标签映射为全局唯一的 L2 Box ID
            local_to_global_l2 = {}
            for idx, label in enumerate(cluster_labels):
                # label == -1 代表噪声点，在实际应用中可以选择归入一个特殊的 "Miscellaneous" Box，或调整聚类参数
                if label not in local_to_global_l2:
                    local_to_global_l2[label] = global_l2_counter
                    global_l2_counter += 1
                
                tools_in_l1[idx]["l2_box_id"] = local_to_global_l2[label]
                processed_data.append(tools_in_l1[idx])
                
        print(f"处理完成！共发现 {global_l2_counter} 个 L2 Boxes。")
        return processed_data

# --- 使用示例 ---
if __name__ == "__main__":
    # 模拟读取 ToolBench 的原始数据
    mock_toolbench_data = [
        {"tool_id": 1, "category_name": "Finance", "tool_name": "get_exchange_rate", "description": "..."},
        {"tool_id": 2, "category_name": "Finance", "tool_name": "calculate_mortgage", "description": "..."},
        {"tool_id": 3, "category_name": "Social", "tool_name": "post_tweet", "description": "..."}
    ]
    
    processor = ToolBenchDataProcessor()
    final_tools = processor.process_and_cluster(mock_toolbench_data)
    
    # 打印其中一个处理后的工具数据
    print(final_tools[0])