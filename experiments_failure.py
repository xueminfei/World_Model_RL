"""P5: forced failure analysis -- how world-model quality (data size) governs Dyna-Q.

Trains the world model on 200 (bad) and 2000 (medium) transitions in addition to
the good 10,000-transition model, freezes each, and runs Dyna-Q (K=10) with it on
the SAME training map, against the K=0 baseline and the Value-Iteration ceiling
(3 seeds, 50,000 real steps). Produces:

  * figure3_learning_curves.png : 4 curves (K=0, Dyna-Q good/medium/bad) + VI
    ceiling, mean +/-1 std -- the shared P4/P5 learning-curve figure.
  * figure5_data_scaling.png    : 50K-step success vs WM data size {200,2000,10000,K=0}.
  * p5_curves.npz               : the aggregated curves.

NOTE (academic-integrity): this script is tooling; the written failure analysis is
the user's own (see report). A timestamped pre-registered prediction lives in
report/p5_prediction_2026-06-15.md, written before this was run.
"""

import os
import time

import numpy as np

from gridworld import GridWorld
from dyna_q import value_iteration, evaluate_greedy
from world_model import train_world_model, RESULTS_DIR
from experiments import load_world_model, SEEDS, N_STEPS, NOISE
from experiments_ablation import run_condition


def ensure_models():
    """Train the 200- and 2000-transition models if not already on disk."""
    for n in (200, 2000):
        tag = f"_n{n}"
        tpath = os.path.join(RESULTS_DIR, f"transition_net{tag}.pt")
        if not os.path.exists(tpath):
            print(f"Training world model on {n} transitions ...")
            train_world_model(n_transitions=n, seed=0, verbose=True)
        else:
            print(f"Found existing {n}-transition model.")


def main(seeds=SEEDS, n_steps=N_STEPS, save=True):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ensure_models()

    # ceiling
    env = GridWorld(noise_prob=NOISE, seed=0)
    P, R = env.get_all_transitions()
    _, Q_vi, _ = value_iteration(P, R)
    vi_success, _ = evaluate_greedy(Q_vi, n_episodes=200, noise_prob=NOISE)
    print(f"[VI] greedy success={vi_success:.2%}")

    good_t, good_r = load_world_model("")           # 10,000
    med_t, med_r = load_world_model("_n2000")       # 2,000
    bad_t, bad_r = load_world_model("_n200")        # 200

    # (label, K, tnet, rnet, color, key)
    conds = [
        ("Q-learning (K=0)",            0,  None,  None,  "#1f77b4", "k0"),
        ("Dyna-Q good (10000)",         10, good_t, good_r, "#d62728", "good"),
        ("Dyna-Q medium (2000)",        10, med_t,  med_r,  "#ff7f0e", "medium"),
        ("Dyna-Q bad (200)",            10, bad_t,  bad_r,  "#9467bd", "bad"),
    ]
    results = []
    for label, K, tn, rn, _, _ in conds:
        print(f"Running {label} x{len(seeds)} seeds ...")
        results.append(run_condition(label, seeds, K, "neural", tn, rn, n_steps))

    if save:
        payload = {"steps": results[0]["steps"], "vi_success": vi_success,
                   "seeds": np.array(seeds)}
        for (label, K, tn, rn, color, key), res in zip(conds, results):
            payload[f"{key}_eval"] = res["eval_success"]
        np.savez(os.path.join(RESULTS_DIR, "p5_curves.npz"), **payload)
        print(f"Saved P5 curves to {RESULTS_DIR}/p5_curves.npz")

    return {"conds": conds, "results": results, "vi_success": vi_success}


if __name__ == "__main__":
    from plotting import plot_learning_curves, plot_data_scaling

    t0 = time.time()
    out = main()

    # Figure 3: the shared 4-line learning curve (overwrites the 2-line P4 draft).
    curves = [{"label": res["label"], "steps": res["steps"],
               "data": res["eval_success"], "color": color}
              for (label, K, tn, rn, color, key), res in zip(out["conds"], out["results"])]
    fig3 = plot_learning_curves(
        curves, vi_ceiling=out["vi_success"],
        title="Figure 3: success rate vs real steps (world-model data size)",
        save_path=os.path.join(RESULTS_DIR, "figure3_learning_curves.png"),
    )
    print(f"Saved {fig3}")

    # Figure 5: 50K-step success vs data size.
    labels, means, stds = [], [], []
    keymap = [("bad", "200"), ("medium", "2000"), ("good", "10000"), ("k0", "K=0")]
    res_by_key = {key: res for (l, K, tn, rn, c, key), res in zip(out["conds"], out["results"])}
    for key, lab in keymap:
        arr = res_by_key[key]["eval_success"][:, -1]  # final (50K) success per seed
        labels.append(lab); means.append(float(arr.mean())); stds.append(float(arr.std()))
    fig5 = plot_data_scaling(labels, means, stds, vi_ceiling=out["vi_success"])
    print(f"Saved {fig5}")
    print(f"Final 50K success: " + ", ".join(f"{l}={m:.0%}" for l, m in zip(labels, means)))
    print(f"Total P5 time: {time.time() - t0:.1f}s")
