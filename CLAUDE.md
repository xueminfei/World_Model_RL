# CLAUDE.md

本文件指导 Claude Code 在本仓库中工作。这是 SJTU《AI / 深度学习与强化学习》课程的**期末项目 Project 10：Teaching an Agent to Imagine**（学习世界模型以提升 RL 样本效率）。

---

## 1. 项目目标（一句话）

在一个有噪声的 8×8 网格世界中，用神经网络学一个**世界模型**（转移网络 + 奖励网络），用它给 **Dyna-Q** 生成"想象的经验"，再把 Dyna-Q 与普通 Q-learning、Value Iteration 对比，回答核心问题：

> **当神经网络能预测环境的下一步时，这种预测真的能让 RL agent 学得更快吗？预测错了又会怎样？**

最终交付：**代码（zip）+ 报告（PDF，约 2300–3400 字，8 节）+ AI 使用日志**。满分 110（缩放到 100）。

---

## 0. 可视化优先原则（贯穿全程）

**这是一个高度依赖可视化的项目**——评分里有 5 张必需图（Figure 1–5），且调试 RL/世界模型几乎只能靠"看图"。因此约定一条工作规则：

> **每当搭建或修改环境/模型后，第一件事是生成并查看可视化图，再继续写下游代码。**

- **环境搭好后**：立刻用 `plotting.plot_grid_layout(env)` 生成 `results/gridworld_layout.png`（墙=灰、起点=绿 S、终点=金 G、格内标 0–63 索引），用 Read 工具看图确认墙的布局合理、起点到终点有通路，再动手写世界模型。
- **每个阶段都先出图再分析**：P1→访问热力图、P2→准确率曲线 + top-3 预测、P3/P4→学习曲线 + 价值热力图、P5→数据量柱状图。
- **动图优先用于"走通"调试**：用 `plotting.animate_episode(env, ...)` 把单个 episode 逐帧导出成 GIF（agent 红点、蓝色轨迹、步数/累计回报、噪声步标红），直观看 agent 怎么动、噪声怎么影响轨迹。后续训练好 agent 后也可传入贪婪策略 `policy=lambda s: argmax(Q[...])` 看学到的行为。
- 所有图/动图存到 `results/`，生成后用 Read 工具实际查看（GIF 用 PIL 抽一帧转 PNG 再看），不要只凭代码假设图是对的。

可视化工具一览（`plotting.py`）：`plot_grid_layout`（静态布局）、`animate_episode`（GIF 动图）、`plot_visitation_heatmap`（Figure 1 访问热力图）。

---

## 2. 技术栈与约定

- **语言/框架**：Python + **PyTorch**（神经网络）、NumPy（Q 表 / Value Iteration / 数据）、Matplotlib（画图）。
- **环境**：纯 CPU 即可，无 GPU 瓶颈（全是 tabular 或小 MLP）。整套实验约 1 小时算力。
- **随机性**：所有实验用固定 seed 列表（如 `[0, 1, 2]`），保证可复现。Q-learning / Dyna-Q 用 **3 个 seeds**。

---

## 3. 文件结构（多文件模块化）

```
World_Model_RL/
├── gridworld.py          # P1: GridWorld 环境
├── utils.py              # state_to_vector / action_to_vector / state_to_index
├── world_model.py        # P2: TransitionNet + RewardNet + 数据采集 + 训练 + 验证
├── dyna_q.py             # P3: dyna_q() / q_learning() / value_iteration()
├── experiments.py        # P4+P5: 主实验、坏模型、数据量扫描，3 seeds
├── plotting.py           # Figure 1–5 的生成
├── main.py               # 入口，按 --part 调度各阶段
├── results/              # 指标 .npz、图 .png、冻结的模型权重 .pt
└── report/               # 报告草稿 + P5 预测（必须带时间戳）
```

各模块对外接口（写代码时遵守这些签名，文档原文如此）：

```python
# gridworld.py
class GridWorld:
    def __init__(self, noise_prob=0.2): ...
    def reset(self): ...            # -> state，整数 tuple (row, col)
    def step(self, action): ...     # -> (next_state, reward, done, info)
                                    #    info = {'true_next': (r,c), 'noisy': bool}
    def get_all_transitions(self): ...  # -> 真实 P[s,a,s'], R[s,a]，仅供 Value Iteration

# utils.py
def state_to_vector(state): ...     # one-hot 64 维
def action_to_vector(action): ...   # one-hot 4 维
# 网络输入 = concat(state_vec, action_vec) = 68 维

# dyna_q.py
def dyna_q(env, transition_model, reward_model, K=10, alpha=0.1, gamma=0.95,
           n_steps=50_000, epsilon_start=1.0, epsilon_end=0.05): ...
def q_learning(env, ...): return dyna_q(env, None, None, K=0, ...)  # K=0 特例
def value_iteration(P, R, gamma=0.95, threshold=1e-6): ...
```

---

## 4. 环境规格（P1，权重 10 分）

- **网格**：8×8，状态用整数 tuple `(row, col)`，row/col ∈ {0..7}，共 **64 个状态**。
- **起点** (0,0) 左上；**终点** (7,7) 右下。
- **墙**：8–10 个固定墙格，设计时**必须保证起点到终点有通路**。
- **动作**：`0=up, 1=down, 2=left, 3=right`（4 个）。
- **随机性 p=0.2**：以 0.2 概率忽略动作、改为均匀随机方向；0.8 执行意图动作。撞墙/出界则原地不动。
- **奖励**：到达 (7,7) +1.0 并结束；其余每步 −0.01（时间惩罚）；100 步未到达则 episode 结束。
- **验证点（Figure 1）**：跑 100 个随机 episode，画 8×8 访问频率热力图——左上应比右下访问更频繁。

> ⚠️ **为什么要随机性**：确定性环境世界模型只是查表，没意思。随机性强迫模型学真正的概率分布，也让 Q-learning 更难，给 Dyna-Q 的想象经验留出帮忙空间。

---

## 5. 世界模型（P2，权重 12 分）

**数据采集**：先用**随机策略**采 **10000 条** transition（覆盖全状态空间，不是只走最优路径！），80/20 划分训练/验证。

**转移网络（分类）**：
```
68 → Linear(256) → ReLU → Linear(256) → ReLU → Linear(64)   # logits over 64 个下一状态
Loss = CrossEntropyLoss，训练 20 epochs
```
验证准确率应达 **75–85%**——**不是 100%**，因为 20% 噪声让部分 transition 本质不可预测。这是环境的根本属性，不是模型缺陷。

**奖励网络（回归）**：
```
68 → Linear(128) → ReLU → Linear(64) → ReLU → Linear(1)
Loss = MSELoss，训练 20 epochs
```
奖励比转移好学（多数 −0.01，只有进 (7,7) 是 +1.0）。

**验证（Figure 2）**：画转移网络验证准确率 vs epoch；并对 5 个 (s,a) 查 top-3 预测下一状态，看是否物理合理。

---

## 6. Dyna-Q（P3，权重 13 分）

Q 表 shape `(64, 4)`。每个真实步后用世界模型生成 **K=10** 个想象步：

```
真实步 → 1 次真实 Q 更新（标准 Q-learning）
       → K 次想象 Q 更新（同一个 Bellman 更新公式）
```

**三个关键设计（必须在报告里解释）**：
1. **想象的 (s,a) 只从 replay buffer 中"访问过的"对里采样**——绝不查询从没见过的状态，否则模型预测不可靠。
2. **世界模型在 RL 开始前就训练好并冻结**，RL 过程中权重不更新（隔离世界模型的纯效应）。
3. **baseline = Dyna-Q(K=0)**（无世界模型）；**天花板 = Value Iteration**（用 `get_all_transitions()` 的真实概率，无需学习，代表最优）。

---

## 7. 实验与图表（P4 + P5）

**主实验（P4）**：3 个 agent（Value Iteration / Q-learning K=0 / Dyna-Q K=10），Q-learning 与 Dyna-Q 各跑 **3 seeds**，每 500 步记录①最近 20 episode 平均奖励 ②成功率。

**强制失败分析（P5，权重 25%，最重要）**：
- 用**只有 200 条** transition 训练一个"坏世界模型"，其余完全一致，跑 200000 步 × 3 seeds → **Dyna-Q (bad model)**。
- 额外训一个 **2000 条**的中等模型做数据量扫描。
- 失败的机制解释要用 Lec18/19/21 概念：坏模型对没见过的 (s,a) 预测近乎随机，想象更新把 Q 值往错误目标拉。

**5 张必需图**（每张要有坐标轴标签、标题、1–2 句 caption）：

| 图 | 内容 | 来源 |
|----|------|------|
| Figure 1 | 随机策略 8×8 访问热力图 | P1 |
| Figure 2 | 转移网络验证准确率 vs epoch | P2 |
| Figure 3 | 学习曲线：成功率 vs 真实步（4 条线，含 VI 虚线天花板，均值±1std 阴影） | P4+P5 |
| Figure 4 | 3 联价值函数热力图（VI / Q-learning / Dyna-Q，同色阶，墙格灰色） | P4 |
| Figure 5 | 50K 步成功率 vs 数据量柱状图（200 / 2000 / 10000 / K=0） | P5 |

---

## 8. 🚨 学术诚信红线（极其重要）

> 文档明确：**实验分析与失败分析的文字推理必须自己写，AI 不能代写。**

Claude 在本项目中**可以**做：
- 写/调试代码（环境、网络、Dyna-Q、画图脚本）。
- 解释概念、定位 bug、优化实现。
- 整理图表、检查计算。

Claude **不能**做（这些会扣分且违反诚信）：
- 代写报告的**实验分析**（P4 的 5 个问题）与**失败分析**（P5 的文字）。
- 代写 **P5 的事前预测**——这必须由人在**跑实验之前**写下并**加时间戳**（截图或脚本顶部带时间的注释），因为那时结果还不存在。
- 报告必须有诚实的 **AI 使用日志**（哪些是 AI 生成、哪些被你修正、生成的代码哪里错了）。

→ 当用户让我写这些分析/预测段落时，应**提醒红线**，转而帮其搭框架、列要点、检查逻辑，由用户自己落笔。

---

## 9. 常见坑（来自文档 Practical Tips）

- **Dyna-Q 表现和 Q-learning 一样** → 通常是：①世界模型没训练好（检查 loss 是否下降）②想象 transition 只从 index 0 的状态采样（检查 replay buffer 采样）③K 被误设为 0。
- **想象步只能采样"访问过的 (s,a)"**——最容易写错成全空间随机采样。
- **Value Iteration** 用 `get_all_transitions()` 的真实 `P[s,a,s']`，64 状态 1 秒内收敛；V* 既做 Figure 3 天花板（200 test episodes 评估贪婪策略），又做 Figure 4 的真值色阶。
- 世界模型**冻结**，别在 RL 循环里误更新它。
- 训练快：20 epochs / 8000 条 < 2 分钟（CPU）；12 个 run（200K × 3 seeds × 4 条件）≈ 1 小时。

---

## 10. 课程对应（写报告引用 lecture 时参考）

| 项目元素 | Lecture |
|----------|---------|
| MDP 形式化 (S,A,R,P,γ) | Lec18 |
| Value Iteration / Bellman 最优 | Lec19 |
| Q-learning 更新规则 / TD error | Lec21 |
| ε-greedy 探索 / 模型化 RL / Dyna-Q | Lec17 |
| 神经网络函数逼近 / 泛化 | Lec08, Lec11 |
| CrossEntropy / Softmax 分类损失 | Lec08, Lec11 |
| one-hot 编码 / 网络输入 | Lec08 |

---

## 11. 报告结构（8 节，约 2300–3400 字）

1. Introduction (200–300) 2. Environment (250–350) 3. World Model (350–450)
4. Dyna-Q Algorithm (300–400) 5. Main Experiment Results (500–700)
6. Failure Analysis (500–700) 7. Reflection (200–300) 8. AI Tool Usage Log (100–200)
