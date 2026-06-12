"""Plotting utilities. Figures are saved under results/.

Visualization-first rule: whenever the environment is built or changed, render
and inspect the gridworld layout before writing any model or RL code.
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

RESULTS_DIR = "results"


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_grid_layout(env, save_path=None, show_indices=True):
    """Render the static gridworld layout: walls, start, goal.

    This is the visualization to generate and inspect right after building the
    environment, before any model or RL code.
    """
    _ensure_results_dir()
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "gridworld_layout.png")

    size = env.SIZE
    grid = np.zeros((size, size))  # 0 = free
    for (r, c) in env.WALLS:
        grid[r, c] = 1  # wall

    fig, ax = plt.subplots(figsize=(6, 6))
    # free = white, wall = dark gray
    ax.imshow(grid, cmap=matplotlib.colors.ListedColormap(["white", "#444444"]),
              vmin=0, vmax=1)

    # start (green) and goal (gold) markers
    sr, sc = env.START
    gr, gc = env.GOAL
    ax.add_patch(plt.Rectangle((sc - 0.5, sr - 0.5), 1, 1, color="#2ca02c", alpha=0.7))
    ax.add_patch(plt.Rectangle((gc - 0.5, gr - 0.5), 1, 1, color="#ffd700", alpha=0.9))
    ax.text(sc, sr, "S", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(gc, gr, "G", ha="center", va="center", fontsize=14, fontweight="bold")

    if show_indices:
        for r in range(size):
            for c in range(size):
                if (r, c) in env.WALLS:
                    continue
                ax.text(c, r + 0.32, f"{env.index((r, c))}", ha="center",
                        va="center", fontsize=6, color="#888888")

    ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax.grid(which="minor", color="#cccccc", linewidth=1)
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.set_title(f"GridWorld layout ({size}x{size}, {len(env.WALLS)} walls, "
                 f"noise={env.noise_prob})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def _base_grid_ax(ax, env):
    """Draw walls / start / goal scaffolding shared by layout and animation."""
    size = env.SIZE
    grid = np.zeros((size, size))
    for (r, c) in env.WALLS:
        grid[r, c] = 1
    ax.imshow(grid, cmap=matplotlib.colors.ListedColormap(["white", "#444444"]),
              vmin=0, vmax=1)
    sr, sc = env.START
    gr, gc = env.GOAL
    ax.add_patch(plt.Rectangle((sc - 0.5, sr - 0.5), 1, 1, color="#2ca02c", alpha=0.7))
    ax.add_patch(plt.Rectangle((gc - 0.5, gr - 0.5), 1, 1, color="#ffd700", alpha=0.9))
    ax.text(sc, sr, "S", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(gc, gr, "G", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax.grid(which="minor", color="#cccccc", linewidth=1)
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xlabel("col")
    ax.set_ylabel("row")


def _rollout(env, policy=None, seed=None, max_steps=None):
    """Run one episode, returning per-step records for animation."""
    if seed is not None:
        env.rng.seed(seed)
    if policy is None:
        policy = lambda s: env.rng.randint(0, env.N_ACTIONS - 1)
    max_steps = max_steps or env.MAX_STEPS

    state = env.reset()
    traj = [{"state": state, "action": None, "reward": 0.0, "noisy": False, "done": False}]
    for _ in range(max_steps):
        action = policy(state)
        next_state, reward, done, info = env.step(action)
        traj.append({"state": next_state, "action": action, "reward": reward,
                     "noisy": info["noisy"], "done": done})
        state = next_state
        if done:
            break
    return traj


def animate_episode(env, policy=None, save_path=None, seed=None, fps=3):
    """Render one episode as an animated GIF: agent position + visited trail.

    Use this during 'walk-through' debugging to watch the agent move step by step
    and to see the effect of the 0.2 transition noise (a red title flags steps
    where the action was overridden by noise).
    """
    _ensure_results_dir()
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "episode_rollout.gif")

    traj = _rollout(env, policy=policy, seed=seed)

    fig, ax = plt.subplots(figsize=(6, 6))

    def draw(i):
        ax.clear()
        _base_grid_ax(ax, env)
        # trail of visited cells up to frame i
        for step in traj[:i + 1]:
            r, c = step["state"]
            if (r, c) not in (env.START, env.GOAL):
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           color="#1f77b4", alpha=0.18))
        # current agent position
        r, c = traj[i]["state"]
        ax.add_patch(plt.Circle((c, r), 0.3, color="#d62728", zorder=5))
        total_r = sum(s["reward"] for s in traj[:i + 1])
        noisy = traj[i]["noisy"]
        title = f"step {i}/{len(traj) - 1}   return={total_r:.2f}"
        if traj[i]["done"]:
            title += "   GOAL!" if traj[i]["reward"] > 0 else "   timeout"
        ax.set_title(title, color="#d62728" if noisy else "black")

    anim = FuncAnimation(fig, draw, frames=len(traj), interval=1000 // fps)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return save_path


def plot_visitation_heatmap(env, n_episodes=100, save_path=None, seed=0):
    """Figure 1: 8x8 state-visitation heatmap under a random policy.

    Counts how often each cell is visited across `n_episodes` random episodes.
    The top-left start region should be visited more than the bottom-right goal;
    cells next to walls slightly less. This verifies the environment behaves.
    """
    _ensure_results_dir()
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "figure1_visitation.png")

    env.rng.seed(seed)
    counts = np.zeros((env.SIZE, env.SIZE))
    for _ in range(n_episodes):
        s = env.reset()
        counts[s] += 1
        for _ in range(env.MAX_STEPS):
            a = env.rng.randint(0, env.N_ACTIONS - 1)
            s, _, done, _ = env.step(a)
            counts[s] += 1
            if done:
                break

    masked = np.ma.masked_where(
        np.array([[(r, c) in env.WALLS for c in range(env.SIZE)]
                  for r in range(env.SIZE)]), counts)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#444444")  # walls in gray
    im = ax.imshow(masked, cmap=cmap)
    for r in range(env.SIZE):
        for c in range(env.SIZE):
            if (r, c) in env.WALLS:
                continue
            ax.text(c, r, int(counts[r, c]), ha="center", va="center",
                    fontsize=7, color="white")
    fig.colorbar(im, ax=ax, label="visit count")
    ax.set_xticks(range(env.SIZE))
    ax.set_yticks(range(env.SIZE))
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.set_title(f"Figure 1: state visitation (random policy, {n_episodes} episodes)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


ACTION_NAMES = ["up", "down", "left", "right"]


def plot_transition_accuracy(acc_history, model, env, sa_pairs=None,
                             save_path=None):
    """Figure 2: transition-net val accuracy vs epoch + top-3 next-state probes.

    Left panel: validation accuracy per epoch, with the 75-85% expectation band
    shaded (it is bounded well below 100% by the env's 20% noise).
    Right panel: for 5 (state, action) pairs, the top-3 predicted next states
    with probabilities, each tagged 'ok' if it is within one step of the queried
    cell (physically plausible) or 'FAR?!' otherwise.
    """
    from world_model import top_k_next  # local import: pulls in torch
    _ensure_results_dir()
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "figure2_transition_accuracy.png")
    if sa_pairs is None:
        # an open-space move, two wall/boundary bumps, a mid-grid move, near goal
        sa_pairs = [((0, 0), 3), ((0, 0), 0), ((4, 4), 1), ((6, 7), 1), ((6, 6), 3)]

    fig, (ax_acc, ax_txt) = plt.subplots(1, 2, figsize=(12, 5),
                                         gridspec_kw={"width_ratios": [1.1, 1]})

    epochs = np.arange(1, len(acc_history) + 1)
    ax_acc.axhspan(0.75, 0.85, color="#2ca02c", alpha=0.12,
                   label="expected band 75-85%")
    ax_acc.plot(epochs, acc_history, marker="o", color="#1f77b4")
    ax_acc.axhline(1.0, color="#888888", ls="--", lw=1)
    ax_acc.text(epochs[-1], 1.0, " 1.0 = unreachable\n (20% noise)",
                va="top", ha="right", fontsize=8, color="#888888")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("validation accuracy")
    ax_acc.set_title("Figure 2: transition-net accuracy vs epoch")
    ax_acc.set_ylim(0, 1.02)
    ax_acc.legend(loc="lower right", fontsize=8)
    ax_acc.grid(alpha=0.3)

    # right panel: top-3 predictions as monospace text
    ax_txt.axis("off")
    ax_txt.set_title("top-3 predicted next state per (s, a)", fontsize=11)
    lines = []
    for state, action in sa_pairs:
        lines.append(f"s={state} idx{env.index(state):<2d}  a={ACTION_NAMES[action]}")
        for idx, prob in top_k_next(model, state, action, k=3):
            nr, nc = env.to_state(idx)
            dist = abs(nr - state[0]) + abs(nc - state[1])
            tag = "ok" if dist <= 1 else "FAR?!"
            wall = " wall" if (nr, nc) in env.WALLS else ""
            lines.append(f"    -> idx{idx:<2d} ({nr},{nc})  p={prob:0.2f}  {tag}{wall}")
        lines.append("")
    ax_txt.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
                family="monospace", fontsize=9, transform=ax_txt.transAxes)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    from gridworld import GridWorld
    layout = plot_grid_layout(GridWorld(seed=0))
    print(f"Saved layout to {layout}")
    heatmap = plot_visitation_heatmap(GridWorld(seed=0))
    print(f"Saved visitation heatmap to {heatmap}")

    # Figure 2 needs a trained model: train (or load) then plot.
    from world_model import train_world_model
    wm = train_world_model(verbose=False)
    fig2 = plot_transition_accuracy(wm["acc_history"], wm["transition_model"],
                                    GridWorld(seed=0))
    print(f"Saved transition-accuracy figure to {fig2}")
    gif = animate_episode(GridWorld(seed=0), seed=42)
    print(f"Saved episode rollout to {gif}")
