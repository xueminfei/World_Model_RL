# 项目进度记录（PROGRESS）

记录已完成的工作，便于下次回顾。最后更新：2026-06-13。

仓库：https://github.com/xueminfei/World_Model_RL

---

## 1. 已完成总览

| 阶段 | 状态 | 产物 |
|------|------|------|
| 仓库 & 文档初始化 | ✅ 完成并已 push | `README.md`、`CLAUDE.md`、原始作业 PDF |
| P1 环境 `gridworld.py` | ✅ 完成并已 push | `gridworld.py` |
| 可视化工具 `plotting.py` | ✅ 完成（Figure 1–4 + 对比动图，部分**本地未 push**） | `plotting.py` |
| 生成的图/动图 | ✅ 已生成（部分**本地未 push**） | `results/`（Fig1–4 + 2 个 GIF + npz）|
| P2 世界模型 | ✅ 完成（本地未 push） | `utils.py`、`world_model.py`、Figure 2 |
| P3 Dyna-Q / Q-learning / Value Iteration | ✅ 完成（本地未 push） | `dyna_q.py` |
| P4 主实验 | ✅ 完成（本地未 push） | `experiments.py`、Figure 3/4、`training_comparison.gif` |
| P5 失败分析 | ⬜ 未开始 | — |
| P6 报告 | ⬜ 完成骨架，分析待人写 | `report/report.tex` |


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

## 2d. `dyna_q.py`（P3 Dyna-Q / Q-learning / Value Iteration，已完成）

三个对外函数严格按 CLAUDE.md §6 签名实现：

- **`value_iteration(P, R, gamma=0.95, threshold=1e-6)`** → `(V, Q, policy)`。用 `get_all_transitions()` 的真实 `P[s,a,s']` 做 Bellman 最优迭代（`Q = R + γ·P·V`，向量化），64 状态秒内收敛。自检：`V(start)=0.330`，贪婪策略 200 episode 成功率 **100%** → 作 Figure 3 天花板 + Figure 4 真值色阶。
- **`dyna_q(env, transition_model, reward_model, K=10, alpha=0.1, gamma=0.95, n_steps=50000, epsilon_start=1.0, epsilon_end=0.05, record_every=500, eval_episodes=0, seed=0)`** → dict（`Q`、`steps`、`avg_reward`、`success_rate`、可选 `eval_success_rate`）。每个真实步：1 次真实 Q 更新 + K 次想象更新。
- **`q_learning(env, ...)`** = `dyna_q(K=0)` 特例，无世界模型 baseline。
- 辅助：`evaluate_greedy(Q, n_episodes=200, ...)` 用独立 env 跑贪婪策略，返回 `(成功率, 平均回报)`，供 Figure 3 天花板/贪婪曲线。

**三个关键设计（CLAUDE.md §6，已落实）**：① 想象的 (s,a) 只从 `visited_list`（真实访问过、去重的 (s,a) 对）采样，绝不查未见状态；② 世界模型 `eval()` + `torch.no_grad()`，RL 循环内权重永不更新；③ 想象的下一状态从转移网络 softmax **采样**（非 argmax，保留学到的随机性），奖励由奖励网络回归；预测到终点(index 63)视为 terminal（不 bootstrap）。K 次想象按 batch 一次前向传播，单线程 `set_num_threads(1)`。

**自检验证（关键，避免"Dyna-Q==Q-learning"坑）**：seed=2、100 eval episode 贪婪成功率：

| 真实步 | K=0 | K=10 |
|--------|-----|------|
| 2500 | 0.06 | **1.00** |
| 5000 | 0.52 | 1.00 |
| 7500 | 0.73 | 1.00 |
| 10000 | 1.00 | 1.00 |

→ 世界模型让 agent 在 ~2500 步就解出环境，比无模型快约 4×，样本效率提升明确可见。整套自检 7 秒（CPU）。

---

## 2e. `experiments.py`（P4 主实验，已完成）

按 CLAUDE.md §7 跑三个 agent，3 seeds=`[0,1,2]`、`n_steps=50000`、`record_every=500`、每检查点 `eval_episodes=30` 贪婪评估：

- `load_world_model(tag)`：加载冻结的转移/奖励网络（tag='' 是 10k 数据模型，P5 用 `_n200`/`_n2000`）。
- `run_condition`：一个条件跨 seeds 跑 `dyna_q`，堆叠每 seed 的学习曲线 + 最终 Q 表。
- `main()`：VI（真实 P,R 求解，作天花板）+ Q-learning(K=0) + Dyna-Q(K=10)，存 `results/p4_curves.npz`（含 steps、各条件 eval_success/avg_reward、vi_success、三个价值函数 V_vi/V_q0/V_qK）。Figure 4 的学习 agent 价值用 **跨 seed 平均 Q 的 max_a**。
- 注：每检查点贪婪评估用**固定 eval seed=99**（在 `dyna_q.py`），曲线反映策略进步而非评估集噪声。

**结果（全套 46 秒，CPU）**：
- **Figure 3**（`plot_learning_curves`，`results/figure3_learning_curves.png`）：成功率 vs 真实步，均值±1std 阴影 + VI 100% 虚线天花板 + "达 50% 成功率所需步数"标注框。**K=10 在 ~2000 步达 50%，K=0 需 ~5500 步（≈2.75× 加速）**，K=10 几乎一上来就贴天花板。已 Read 确认。函数设计成可加任意条曲线（P5 坏模型线直接塞进同图）。
- **Figure 4**（`plot_value_heatmaps`，`results/figure4_value_heatmaps.png`）：VI / K=0 / K=10 三联价值热力图，**共享色阶**、墙灰、终点=0（吸收态）。**Dyna-Q 的 V(s) 全程贴近 V\***（如第 0 行 0.31–0.59 vs VI 0.33–0.60）；**Q-learning 在少访问区严重低估**（右上角 0.05–0.20 vs VI 0.43–0.55）→ 直观展示"想象更新把价值传播到 agent 很少踏足的状态"。已 Read 确认。

> Figure 3/4 的 caption 文字与 §5 主实验五问分析仍是**人写**（CLAUDE.md §8 红线），AI 只出图与事实。

**训练过程对比动图（CLAUDE.md §0 要求，已完成）** → `results/training_comparison.gif`：
- `dyna_q` 新增 `snapshot_steps` 参数：在指定真实步数处存一份 `Q.copy()` 到 `out["snapshots"]`（`q_learning` 也透传）。
- `plotting.animate_training_comparison(env, snapshots_a, snapshots_b, checkpoints, ...)`：左 K=0 / 右 K=10 双面板，遍历训练预算 `[1000,2000,3500,5000,8000,12000]`，每个预算把两个 agent 当时冻结的**贪婪策略**在**同一个实环境（同噪声序列，`env.rng.seed` 对齐）**里逐步走，lockstep 并排播放。到终点标题变绿 GOAL，超时标 timeout。
- `experiments.make_training_comparison_gif()` 一键复现（已并入 `experiments.py` 的 `__main__`）。
- **直观证据**（贪婪 rollout 步数，同 seed）：预算 2000–3500 步时 **K=10 用 21 步到终点、K=0 撞满 40 步超时**；5000–8000 步 K=10 已稳定到终点、K=0 轨迹仍困在左上；12000 步两者都到终点（return 0.95）。已抽帧拼图 Read 确认。这正是"同样真实步数下，有世界模型让 agent 更快学会在真环境里走对"的可视化。

---

## 3. `plotting.py`（可视化工具）

所有图存到 `results/`。函数一览（P1/P2 → P3/P4）：

1. **`plot_grid_layout(env)`** → `results/gridworld_layout.png`
   静态布局图：墙=灰、起点=绿 S、终点=金 G、每格标 0–63 索引。**搭好环境后第一件事就是看它**，确认墙布局合理、有通路。

2. **`animate_episode(env, policy=None, seed=None, fps=3)`** → `results/episode_rollout.gif`
   把单个 episode 逐帧导出成 GIF，用于"走通"调试：
   - 红点 = agent 当前位置；浅蓝格 = 走过的轨迹；
   - 标题显示 `step i/N`、累计 return；**触发噪声的步标题变红**；结束时显示 GOAL/timeout。
   - `policy=None` 默认随机策略；训练出 Q 表后可传贪婪策略看学到的行为。

3. **`plot_visitation_heatmap(env, n_episodes=100)`** → `results/figure1_visitation.png`（**Figure 1**）
   随机策略跑 100 个 episode，统计每格访问次数画热力图。墙=灰。用于验证环境：左上起点区访问最多，右下终点区最少。

4. **`plot_transition_accuracy(acc_history, model, env, ...)`** → `results/figure2_transition_accuracy.png`（**Figure 2**）
   左：转移网络验证准确率 vs epoch（含 75–85% 期望带）；右：5 个 (s,a) 的 top-3 预测下一状态 + 物理合理性标注。

5. **`plot_learning_curves(curves, vi_ceiling=None, ...)`** → `results/figure3_learning_curves.png`（**Figure 3**）
   成功率 vs 真实步，均值±1std 阴影、VI 天花板虚线、"达阈值所需步数"标注框。`curves` 是可变长列表，P5 坏模型线可直接加进同图。

6. **`plot_value_heatmaps(env, value_maps, ...)`** → `results/figure4_value_heatmaps.png`（**Figure 4**）
   多面板价值热力图，共享色阶、墙灰、终点=0（吸收态）。

7. **`animate_training_comparison(env, snapshots_a, snapshots_b, checkpoints, ...)`** → `results/training_comparison.gif`
   左 K=0 / 右 K=10 双面板，遍历训练预算、同噪声实环境里 lockstep 走贪婪策略，看"同步数下谁先到终点"。

---

## 4. 已生成的可视化产物（`results/`）

- **`gridworld_layout.png`** — 静态布局，确认无误（S 左上、G 右下、10 墙、有通路）。
- **`figure1_visitation.png`（Figure 1）** — 访问热力图，确认无误：起点 (0,0) 最亮（869 次），平滑衰减到终点 (7,7) 最暗（7 次），墙为灰。
- **`figure2_transition_accuracy.png`（Figure 2）** — 转移网络准确率 87.45% + 5 个 top-3 预测全部物理合理。
- **`figure3_learning_curves.png`（Figure 3）** — K=10 ~2000 步达 50%、K=0 ~5500 步（≈2.75× 加速），均贴 VI 100% 天花板。已 Read 确认。
- **`figure4_value_heatmaps.png`（Figure 4）** — Dyna-Q 的 V(s) 全程贴近 V\*，Q-learning 在少访问区严重低估。已 Read 确认。
- **`training_comparison.gif`** — 训练过程对比动图（K=0 vs K=10，6 个预算 × 同噪声实环境贪婪 rollout），206 帧。已抽帧拼图 Read 确认。
- **`p4_curves.npz`** — P4 聚合曲线 + 三个价值函数，供报告/再绘图。
- **`episode_rollout.gif`** — 随机策略单 episode 回放（调试用，非交付图）。

---

## 5. `CLAUDE.md` 关键约定（已写入）

- **Section 0 可视化优先原则**：这是高度依赖可视化的项目；每次搭/改环境或模型后，第一件事是生成并用 Read 工具查看相应图，再写下游代码；动图优先用于走通调试。
- **学术诚信红线**：实验分析（P4 五问）、失败分析（P5）、P5 事前预测，AI **不能代写**；必须人写，且 P5 预测要在跑实验前加时间戳。报告需诚实的 AI 使用日志。
- 文件结构、各模块接口签名、环境/世界模型/Dyna-Q 规格、5 张必需图、常见坑、课程 lecture 对应表、报告 8 节结构。

---

## 6. 技术栈与运行

- Python3 + PyTorch + NumPy + Matplotlib + Pillow（GIF）。纯 CPU，全套实验约 1 小时。
- 运行环境自检：`python3 gridworld.py`
- 世界模型训练：`python3 world_model.py`（~6 秒）
- Dyna-Q 自检：`python3 dyna_q.py`（~7 秒，含 VI/K=0/K=10 对比）
- P4 主实验 + Figure 3/4 + 对比动图：`python3 experiments.py`（~50 秒 + 动图 ~40 秒）
- 静态可视化（Figure 1/2 等）：`python3 plotting.py`

---

## 7. 下一步

1. （可选）把 P3/P4 的本地改动（`dyna_q.py`、`experiments.py`、`plotting.py` 新增函数、Figure 3/4、对比动图、PROGRESS 更新）提交并 push。
2. **P5 失败分析**（权重 25%，最重要）：
   - 🚨 **先**写 P5 事前预测并**加时间戳**（跑实验之前，人写，AI 不代写）。
   - 用 `world_model.train_world_model(n_transitions=200)` / `=2000` 训坏模型/中模型（带 tag 存权重）。
   - 跑 Dyna-Q(bad model) 200000 步 × 3 seeds，把坏模型线加进 **Figure 3**（`plot_learning_curves` 已支持可变长 curves）。
   - **Figure 5**：50K 步成功率 vs 数据量柱状图（200 / 2000 / 10000 / K=0）——需在 `plotting.py` 新增 `plot_data_sweep`。
3. **P6 报告**：填 §5 主实验五问、§6 失败分析、§7 反思、§8 AI 日志（全部人写），取消 report.tex 里 Figure 3–5 的 includegraphics 注释。
