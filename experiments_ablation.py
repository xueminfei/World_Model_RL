"""Imagination-source ablation: where should the K imagined updates come from?

The project's main comparison (Q-learning K=0 vs neural Dyna-Q K=10) moves TWO
knobs at once -- it turns the imagination loop on AND supplies a learned neural
world model. This script holds the imagination loop fixed at K=10 and varies only
the *source* of the imagined (s', r), isolating what the neural world model adds
over cheaper alternatives:

  * Q-learning (K=0)        : no imagination at all -- the floor.
  * Replay (K=10)           : experience replay. K real, previously-seen
                              transitions reused. No model -- imagining REAL data.
  * Tabular Dyna-Q (K=10)   : classic Sutton Dyna-Q. A lookup table memorises the
                              last (s', r) per visited (s,a). A model, but not a
                              learned/generalising one -- imagining MEMORISED data.
  * Neural Dyna-Q (K=10)    : the project's agent. The frozen MLP world model
                              predicts (s', r) and can generalise -- imagining
                              GENERATED data.

All four share the same Bellman update, alpha, gamma, epsilon schedule, seeds and
step budget; only `imagine=` differs. Curves go to results/ablation_curves.npz and
results/figure_ablation_imagination.png.

NOTE (academic-integrity): this script and its figure are tooling. The written
interpretation of *why* the sources rank the way they do is yours to write.
"""

import os
import time

import numpy as np

from gridworld import GridWorld
from dyna_q import dyna_q, value_iteration, evaluate_greedy
from experiments import load_world_model, SEEDS, N_STEPS, RECORD_EVERY, EVAL_EPISODES, NOISE


def run_condition(label, seeds, K, imagine, tnet, rnet, n_steps=N_STEPS):
    """Run one imagination-source condition across seeds; stack per-seed curves."""
    evals, finalQ = [], []
    steps = None
    for seed in seeds:
        env = GridWorld(noise_prob=NOISE, seed=seed)
        res = dyna_q(env, tnet, rnet, K=K, n_steps=n_steps, imagine=imagine,
                     record_every=RECORD_EVERY, eval_episodes=EVAL_EPISODES, seed=seed)
        steps = res["steps"]
        evals.append(res["eval_success_rate"])
        finalQ.append(res["Q"])
        print(f"  [{label}] seed={seed}: final greedy success={res['eval_success_rate'][-1]:.2%}")
    return {"label": label, "steps": steps,
            "eval_success": np.array(evals), "final_Q": np.array(finalQ)}


def main(seeds=SEEDS, n_steps=N_STEPS, save=True):
    from world_model import RESULTS_DIR

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Ceiling.
    env = GridWorld(noise_prob=NOISE, seed=0)
    P, R = env.get_all_transitions()
    _, Q_vi, _ = value_iteration(P, R)
    vi_success, _ = evaluate_greedy(Q_vi, n_episodes=200, noise_prob=NOISE)
    print(f"[VI] greedy success={vi_success:.2%}")

    tnet, rnet = load_world_model()

    conds = [
        ("Q-learning (K=0)",      0,  "neural",  "#1f77b4"),
        ("Replay (K=10)",         10, "replay",  "#9467bd"),
        ("Tabular Dyna-Q (K=10)", 10, "tabular", "#ff7f0e"),
        ("Neural Dyna-Q (K=10)",  10, "neural",  "#d62728"),
    ]
    results = []
    for label, K, imagine, _ in conds:
        print(f"Running {label} x{len(seeds)} seeds ...")
        results.append(run_condition(label, seeds, K, imagine, tnet, rnet, n_steps))

    if save:
        payload = {"steps": results[0]["steps"], "vi_success": vi_success,
                   "seeds": np.array(seeds)}
        for (label, K, imagine, _), res in zip(conds, results):
            key = "k0" if K == 0 else imagine  # k0 / replay / tabular / neural
            payload[f"{key}_eval"] = res["eval_success"]
        np.savez(os.path.join(RESULTS_DIR, "ablation_curves.npz"), **payload)
        print(f"Saved ablation curves to {RESULTS_DIR}/ablation_curves.npz")

    return {"conds": conds, "results": results, "vi_success": vi_success}


if __name__ == "__main__":
    from plotting import plot_learning_curves

    t0 = time.time()
    out = main()

    curves = [{"label": res["label"], "steps": res["steps"],
               "data": res["eval_success"], "color": color}
              for (label, K, imagine, color), res in zip(out["conds"], out["results"])]
    fig = plot_learning_curves(
        curves, vi_ceiling=out["vi_success"],
        title="Imagination-source ablation: K=10 with different (s', r) sources",
        save_path=os.path.join("results", "figure_ablation_imagination.png"),
    )
    print(f"Saved {fig}")
    print(f"Total ablation time: {time.time() - t0:.1f}s")
