# 项目进度记录（PROGRESS）

记录已完成的工作，便于下次回顾。最后更新：2026-06-12。

仓库：https://github.com/xueminfei/World_Model_RL

---

## 1. 已完成总览

| 阶段 | 状态 | 产物 |
|------|------|------|
| 仓库 & 文档初始化 | ✅ 完成并已 push | `README.md`、`CLAUDE.md`、原始作业 PDF |
| P1 环境 `gridworld.py` | ✅ 完成并已 push | `gridworld.py` |
| 可视化工具 `plotting.py` | ✅ 完成（动图+热力图部分**本地未 push**，等 review） | `plotting.py` |
| 生成的图/动图 | ✅ 已生成（部分**本地未 push**） | `results/` 下 3 个文件 |
| P2 世界模型 | ✅ 完成（本地未 push） | `utils.py`、`world_model.py`、Figure 2 |
| P3 Dyna-Q / Q-learning / Value Iteration | ⬜ 未开始 | — |
| P4 主实验 | ⬜ 未开始 | — |
| P5 失败分析 | ⬜ 未开始 | — |
| P6 报告 | ⬜ 未开始 | — |

> Git 状态提醒：`gridworld.py`、`plot_grid_layout`、布局图、CLAUDE.md Section 0 已 push。
> **`animate_episode` / `plot_visitation_heatmap` 两个函数、对应的 GIF 和 Figure 1、CLAUDE.md 动图说明，目前还在本地未提交**（用户正在 review）。

---

## 2. `gridworld.py`（P1 环境，已完成）

`GridWorld` 类，严格按作业规格实现：

- **8×8 网格**，状态 `(row, col)`，共 64 个状态；`index()` / `to_state()` 在整数索引 0–63 与坐标间转换。
- **起点** (0,0)，**终点** (7,7)。
- **10 个固定墙格**：`(1,1),(1,2),(1,3),(2,6),(3,3),(3,4),(3,5),(4,6),(5,1),(5,2)`；`__init__` 里用 BFS `_path_exists()` 断言起点到终点有通路。
- **动作** `0=up,1=down,2=left,3=right`；撞墙/出界则原地不动。
- **随机性 `noise_prob=0.2`**：0.8 执行意图动作，0.2 改为均匀随机方向。每个 env 实例有独立的 `random.Random(seed)`，可复现。
- **奖励**：到 (7,7) +1.0 且 `done`；其余每步 −0.01；100 步未到达则结束。
- **`step()` 返回** `(next_state, reward, done, info)`，`info = {'true_next': (r,c), 'noisy': bool}`。`true_next` = 无噪声时意图动作会到的格子；`noisy` = 这一步是否触发了噪声分支。
- **`get_all_transitions()`** 返回真实 `P[s,a,s']`（64,4,64）和 `R[s,a]`（64,4），仅供 Value Iteration。终点设为吸收态（自环、零奖励），保证 VI 中 V(goal)=0。
- 文件底部 `__main__` 有自检：验证 P 每行和为 1、通路存在、随机策略 100 episode 的到达率。

**自检结果**：P 行和=1 ✅；通路存在 ✅；随机策略到达率约 **7/100**（这个环境对随机策略确实很难，符合预期）。

---

## 2b. `utils.py` + `world_model.py`（P2 世界模型，已完成）

- **`utils.py`**：`state_to_index` / `index_to_state` / `state_to_vector`(64 one-hot) / `action_to_vector`(4 one-hot) / `sa_to_vector`(68 维网络输入)。
- **`world_model.py`**：
  - `collect_transitions`：随机策略采 10000 条 `(s,a,s',r)`（覆盖全状态空间），`done` 即 reset 续采。
  - `train_val_split`：80/20。
  - `TransitionNet` 68→256→256→64（CrossEntropy）、`RewardNet` 68→128→64→1（MSE），各 20 epochs，Adam。
  - `train_world_model`：一键采集+训练+保存冻结权重(`transition_net.pt`/`reward_net.pt`)和 metrics(`world_model_metrics.npz`)。坏模型/数据量扫描用 `n_transitions=200/2000` 带 tag 存。
  - ⚠️ **坑已修**：torch 默认对小 batch 开满 CPU 线程，反而烧满 ~30 核空转（实测 184 CPU-min 没跑完）。加 `torch.set_num_threads(1)` 后 **6 秒** 跑完。
- **结果**：转移网络验证准确率 **87.45%**（略高于作业 75–85% 带，因墙角方向塌缩使天花板>85%，已在报告 §3 解释；关键点"非 100%、20% 噪声决定"成立）；奖励 MSE 9.4e-5。
- **Figure 2**（`plotting.plot_transition_accuracy`）：准确率 vs epoch（含期望带+1.0 虚线）+ 5 个 (s,a) top-3 预测，全部物理合理(标 ok)。已生成 `results/figure2_transition_accuracy.png` 并 Read 确认。

## 2c. `report/report.tex`（报告骨架，已编译）

- 8 节结构齐全，`\today` 日期，`\todo{}` 红框标出**必须人写**的部分。
- §1–4（Intro/Env/World Model/Dyna-Q 方法）已填事实性草稿；§3 含 87% 的诚实解释。
- 🚨 §5 主实验分析(5 问)、§6 失败分析+**P5 事前预测(需跑实验前加时间戳)**、§7 反思、§8 AI 日志 —— 全部留 `\todo` 占位，**AI 不代写**（CLAUDE.md §8 红线）。
- Figure 3–5 的 `\includegraphics` 已注释（P4/P5 生成后取消注释）。`pdflatex` 编译通过 → `report/report.pdf`。

## 3. `plotting.py`（可视化工具）

所有图存到 `results/`。三个函数：

1. **`plot_grid_layout(env)`** → `results/gridworld_layout.png`
   静态布局图：墙=灰、起点=绿 S、终点=金 G、每格标 0–63 索引。**搭好环境后第一件事就是看它**，确认墙布局合理、有通路。

2. **`animate_episode(env, policy=None, seed=None, fps=3)`** → `results/episode_rollout.gif`
   把单个 episode 逐帧导出成 GIF，用于"走通"调试：
   - 红点 = agent 当前位置；浅蓝格 = 走过的轨迹；
   - 标题显示 `step i/N`、累计 return；**触发噪声的步标题变红**；结束时显示 GOAL/timeout。
   - `policy=None` 默认随机策略；训练出 Q 表后可传贪婪策略看学到的行为。

3. **`plot_visitation_heatmap(env, n_episodes=100)`** → `results/figure1_visitation.png`（**Figure 1**）
   随机策略跑 100 个 episode，统计每格访问次数画热力图。墙=灰。用于验证环境：左上起点区访问最多，右下终点区最少。

---

## 4. 已生成的可视化产物（`results/`）

- **`gridworld_layout.png`** — 静态布局，确认无误（S 左上、G 右下、10 墙、有通路）。
- **`figure1_visitation.png`（Figure 1）** — 访问热力图，确认无误：起点 (0,0) 最亮（869 次），平滑衰减到终点 (7,7) 最暗（7 次），墙为灰。验证环境行为正确。
- **`episode_rollout.gif`** — 随机策略单 episode 回放（seed=42，70 帧）。这次随机走没到终点，第 69 步超时结束（与 Figure 1 右下角访问极少一致）。属于调试用，非交付图。

---

## 5. `CLAUDE.md` 关键约定（已写入）

- **Section 0 可视化优先原则**：这是高度依赖可视化的项目；每次搭/改环境或模型后，第一件事是生成并用 Read 工具查看相应图，再写下游代码；动图优先用于走通调试。
- **学术诚信红线**：实验分析（P4 五问）、失败分析（P5）、P5 事前预测，AI **不能代写**；必须人写，且 P5 预测要在跑实验前加时间戳。报告需诚实的 AI 使用日志。
- 文件结构、各模块接口签名、环境/世界模型/Dyna-Q 规格、5 张必需图、常见坑、课程 lecture 对应表、报告 8 节结构。

---

## 6. 技术栈与运行

- Python3 + PyTorch（后续网络）+ NumPy + Matplotlib + Pillow（GIF）。纯 CPU，全套实验约 1 小时。
- 运行环境自检：`python3 gridworld.py`
- 生成全部可视化：`python3 plotting.py`

---

## 7. 下一步

1. （可选）把当前本地未提交的动图/热力图相关改动提交并 push。
2. **P2 世界模型**：`world_model.py` —— 随机策略采 10000 条 transition、80/20 划分、训练转移网络（CrossEntropy，68→256→256→64）和奖励网络（MSE，68→128→64→1），各 20 epochs，画 Figure 2（验证准确率 vs epoch）+ 5 个 (s,a) 的 top-3 预测。
3. P3 Dyna-Q / Q-learning / Value Iteration → P4 主实验 → P5 失败分析 → P6 报告。
