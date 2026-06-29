"""Generalization test: do the frozen world models transfer to a NEW map?

Both world models are trained on the ORIGINAL map (P2) and FROZEN. Here we drop
them onto a different wall layout (same 8x8, same start (0,0) and goal (7,7),
~same number of walls) and run Dyna-Q without retraining the models. Four curves:

  * Value Iteration         : oracle on the NEW map (true P, R) -- ceiling.
  * Tabular Dyna-Q          : no transfer; builds its model-table from its own
                              new-map experience (learns the new map from scratch).
  * World Model Dyna-Q      : the one-hot WM, frozen from the old map. Its input is
                              only the absolute index, so it replays OLD-map
                              transitions -> expected to transfer poorly.
  * Improved WM Dyna-Q      : the improved WM (one-hot anchor + 3x3 window), frozen
                              from the old map. It additionally sees the NEW map's
                              local walls through the window. Does that extra local
                              input buy any generalization over the one-hot WM?

The honest expectation (predicting an ABSOLUTE next state cannot fully transfer):
both world models may do poorly; we run it anyway to compare them head-to-head.

NOTE (academic-integrity): this script is tooling. The written interpretation of
whether/why the improved model transfers better is the user's to write.
"""

import os
import time

import numpy as np

from gridworld import GridWorld
from dyna_q import value_iteration, evaluate_greedy
from experiments import load_world_model, SEEDS, N_STEPS, NOISE
from experiments_ablation import run_condition
from world_model import load_improved_wm, improved_adapters, RESULTS_DIR
from utils import obs_batch_vectors, build_obs_table

# A NEW, fixed wall layout (different from GridWorld.WALLS), ~11 walls, same start
# (0,0) and goal (7,7). GridWorld.__init__ asserts a start->goal path exists.
NEW_WALLS = frozenset({
    (0, 4), (1, 4), (2, 4),       # upper vertical barrier on col 4
    (4, 1), (4, 2), (4, 3),       # middle horizontal barrier
    (3, 6), (4, 6), (5, 6),       # right vertical barrier on col 6
    (6, 3), (6, 4),               # lower horizontal stub
})
NEW_GOAL = GridWorld.GOAL         # (7, 7), unchanged
NEW_START = GridWorld.START       # (0, 0), unchanged


def make_env(seed):
    return GridWorld(noise_prob=NOISE, seed=seed, walls=NEW_WALLS)


def main(seeds=SEEDS, n_steps=N_STEPS, save=True):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Visualise the new map first (visualization-first rule).
    from plotting import plot_grid_layout
    layout = plot_grid_layout(make_env(0),
                              save_path=os.path.join(RESULTS_DIR, "newmap_layout.png"))
    print(f"Saved new-map layout to {layout}")

    # --- ceiling: Value Iteration on the NEW map --------------------------- #
    env0 = make_env(0)
    P, R = env0.get_all_transitions()
    _, Q_vi, _ = value_iteration(P, R)
    vi_success, _ = evaluate_greedy(Q_vi, n_episodes=200, noise_prob=NOISE,
                                    walls=NEW_WALLS, goal=NEW_GOAL, start=NEW_START)
    print(f"[VI on new map] greedy success={vi_success:.2%}")

    # --- frozen world models (trained on the OLD map) ---------------------- #
    tnet, rnet = load_world_model()                      # one-hot baseline WM
    imp_t, imp_r = improved_adapters(load_improved_wm())  # improved WM (p=0)
    new_table = build_obs_table(walls=NEW_WALLS, goal=NEW_GOAL)
    improved_encoder = lambda s, a: obs_batch_vectors(s, a, obs_table=new_table)

    # (label, K, imagine, color, tnet, rnet, sa_encoder, key)
    conds = [
        ("Tabular Dyna-Q (K=10)",              10, "tabular", "#ff7f0e", None,  None,  None,             "tabular"),
        ("World Model Dyna-Q (K=10)",          10, "neural",  "#d62728", tnet,  rnet,  None,             "neural"),
        ("Improved World Model Dyna-Q (K=10)", 10, "neural",  "#2ca02c", imp_t, imp_r, improved_encoder, "improved"),
    ]
    results = []
    for label, K, imagine, _, tn, rn, enc, _ in conds:
        print(f"Running {label} on new map x{len(seeds)} seeds ...")
        results.append(run_condition(label, seeds, K, imagine, tn, rn, n_steps,
                                     sa_encoder=enc, env_fn=make_env))

    if save:
        payload = {"steps": results[0]["steps"], "vi_success": vi_success,
                   "seeds": np.array(seeds)}
        for (label, K, imagine, _, _, _, _, key), res in zip(conds, results):
            payload[f"{key}_eval"] = res["eval_success"]
        np.savez(os.path.join(RESULTS_DIR, "newmap_curves.npz"), **payload)
        print(f"Saved new-map curves to {RESULTS_DIR}/newmap_curves.npz")

    return {"conds": conds, "results": results, "vi_success": vi_success}


if __name__ == "__main__":
    from plotting import plot_learning_curves

    t0 = time.time()
    out = main()
    curves = [{"label": res["label"], "steps": res["steps"],
               "data": res["eval_success"], "color": color}
              for (label, K, imagine, color, *_), res in zip(out["conds"], out["results"])]
    fig = plot_learning_curves(
        curves, vi_ceiling=out["vi_success"],
        title="New-map generalization: frozen world models transferred to an unseen map",
        save_path=os.path.join(RESULTS_DIR, "figure_newmap_generalization.png"),
    )
    print(f"Saved {fig}")
    print(f"Total new-map time: {time.time() - t0:.1f}s")
