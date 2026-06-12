"""Plotting utilities. Figures are saved under results/.

Visualization-first rule: whenever the environment is built or changed, render
and inspect the gridworld layout before writing any model or RL code.
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    from gridworld import GridWorld
    path = plot_grid_layout(GridWorld(seed=0))
    print(f"Saved layout to {path}")
