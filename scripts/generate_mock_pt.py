#!/usr/bin/env python3
"""
生成 mock 集群参数 .pt 文件。
用于离线测试，数据为随机初始化（维度、形状与真实数据一致）。

运行：
    python scripts/generate_mock_pt.py

输出到：
    mock_data/base/clusters/*.pt
    mock_data/task1/clusters/*.pt
"""
import os, torch, sys

# 确保能 import 项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- Base 集群参数 ----
# 8 个工具，ID 0~7，8 个 L2 盒子（每工具一个盒子），维度 384（L2 用），128（L1 用）
DIM_EMB = 384
NUM_TOOLS_BASE = 8
NUM_L2_BASE = 8
NUM_L1 = 4

# 模拟：3 个 L1 盒子，ID 0~2
l1_centers = torch.randn(NUM_L1, DIM_EMB) * 0.5
l1_widths  = torch.ones(NUM_L1, DIM_EMB) * 0.8

# L2 盒子中心（每工具一个盒子，ID 0~7）
l2_centers = torch.randn(NUM_L2_BASE, DIM_EMB) * 0.3
l2_widths  = torch.ones(NUM_L2_BASE, DIM_EMB) * 0.4

# tool_to_l2: 每个工具属于哪个 L2 盒子（直接 1:1）
tool_to_l2 = {i: i for i in range(NUM_TOOLS_BASE)}

# l2_to_l1: 每个 L2 盒子属于哪个 L1（均匀分配）
l2_to_l1 = {i: i % NUM_L1 for i in range(NUM_L2_BASE)}

# m_global: 转移矩阵 [num_tools, num_boxes]
m_global = torch.zeros(NUM_TOOLS_BASE, NUM_L2_BASE)
for t in range(NUM_TOOLS_BASE):
    box = tool_to_l2.get(t, 0)
    m_global[t, box] = 1.0

# 存盘
BASE_CL = "mock_data/base/clusters"
TASK1_CL = "mock_data/task1/clusters"
os.makedirs(BASE_CL, exist_ok=True)
os.makedirs(TASK1_CL, exist_ok=True)

torch.save(l1_centers, f"{BASE_CL}/l1_centers.pt")
torch.save(l1_widths,  f"{BASE_CL}/l1_widths.pt")
torch.save(l2_centers, f"{BASE_CL}/l2_centers.pt")
torch.save(l2_widths,  f"{BASE_CL}/l2_widths.pt")
torch.save(tool_to_l2, f"{BASE_CL}/tool_to_l2.pt")
torch.save(l2_to_l1,   f"{BASE_CL}/l2_to_l1.pt")
torch.save(m_global,    f"{BASE_CL}/m_global.pt")
print(f"✅ Base clusters → {BASE_CL}/")

# ---- Task1 增量集群参数 ----
# 在 Base 基础上新增 3 个工具，ID 11112~11114，新增 3 个 L2 盒子
# 总共：10 个 L2 盒子，ID 0~9
NUM_TOOLS_TASK1 = NUM_TOOLS_BASE + 3
NUM_L2_TASK1 = NUM_L2_BASE + 3
OLD_NUM_L2 = NUM_L2_BASE
OLD_NUM_TOOLS = NUM_TOOLS_BASE

# 新盒子中心：在旧盒子附近稍微偏移（模拟增量学习）
new_l2_centers = l2_centers[:3] + torch.randn(3, DIM_EMB) * 0.2
new_l2_widths  = torch.ones(3, DIM_EMB) * 0.4

combined_l2_centers = torch.cat([l2_centers, new_l2_centers], dim=0)
combined_l2_widths  = torch.cat([l2_widths,  new_l2_widths],  dim=0)

# 扩展 tool_to_l2
tool_to_l2_t1 = dict(tool_to_l2)
for i, tid in enumerate(range(OLD_NUM_TOOLS, OLD_NUM_TOOLS + 3)):
    tool_to_l2_t1[tid] = OLD_NUM_L2 + i

# 扩展 l2_to_l1
l2_to_l1_t1 = dict(l2_to_l1)
for i in range(3):
    l2_to_l1_t1[OLD_NUM_L2 + i] = (OLD_NUM_L2 + i) % NUM_L1

# 扩展 m_global
m_global_t1 = torch.zeros(NUM_TOOLS_TASK1, NUM_L2_TASK1)
m_global_t1[:OLD_NUM_TOOLS, :OLD_NUM_L2] = m_global[:OLD_NUM_TOOLS, :OLD_NUM_L2]
# 新工具：新盒子行 = 均匀先验
for t in range(OLD_NUM_TOOLS, NUM_TOOLS_TASK1):
    m_global_t1[t, :] = 1.0 / NUM_L2_TASK1

# 复用 base 的 L1 中心/宽度
torch.save(combined_l2_centers, f"{TASK1_CL}/l2_centers.pt")
torch.save(combined_l2_widths,  f"{TASK1_CL}/l2_widths.pt")
torch.save(tool_to_l2_t1,       f"{TASK1_CL}/tool_to_l2.pt")
torch.save(l2_to_l1_t1,         f"{TASK1_CL}/l2_to_l1.pt")
torch.save(m_global_t1,         f"{TASK1_CL}/m_global.pt")

# task1_tools_with_id.json
import json
task1_tools_with_id = {
    str(11112): "Tool: StockAPI. Description: 股票行情. API: quote_stock. API Description: 获取股票报价",
    str(11113): "Tool: SportsAPI. Description: 体育比分. API: get_scores. API Description: 查询体育赛事比分",
    str(11114): "Tool: TravelAPI. Description: 机票预订. API: book_flight. API Description: 预订航班机票",
}
with open(f"{TASK1_CL}/task1_tools_with_id.json", "w", encoding="utf-8") as f:
    json.dump(task1_tools_with_id, f, ensure_ascii=False, indent=2)

print(f"✅ Task1 clusters → {TASK1_CL}/")
print(f"   L2: {NUM_L2_BASE} → {NUM_L2_TASK1}, Tools: {NUM_TOOLS_BASE} → {NUM_TOOLS_TASK1}")
print(f"✅ mock .pt 文件生成完毕！")
