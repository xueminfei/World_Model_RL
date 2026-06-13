"""P4: the main experiment -- Value Iteration vs Q-learning vs Dyna-Q.

Runs the three learners on the same environment and records learning curves so
Figure 3 (success rate vs real steps) and Figure 4 (value-function heatmaps) can
be drawn:

  * Value Iteration : oracle ceiling, solved once from the true (P, R).
  * Q-learning (K=0): no-model baseline, 3 seeds.
  * Dyna-Q (K=10)   : world-model agent, 3 seeds.

The frozen transition / reward nets from world_model.py (P2) are loaded and
reused unchanged. Aggregated curves and the final value functions are saved to
results/p4_curves.npz for plotting and the report.

(P5 -- the bad-model and data-sweep failure analysis -- builds on this file and
slots extra lines into the same Figure 3; that is a separate run.)
"""

import os

import numpy as np
import torch

from gridworld import GridWorld
from world_model import TransitionNet, RewardNet, RESULTS_DIR
from dyna_q import dyna_q, q_learning, value_iteration, evaluate_greedy

SEEDS = [0, 1, 2]
N_STEPS = 50_000
RECORD_EVERY = 500
EVAL_EPISODES = 30  # greedy-eval episodes per checkpoint (clean Figure 3 curve)
NOISE = 0.2


def load_world_model(tag=""):
    """Load a frozen (transition, reward) net pair. tag='' is the 10k-data model."""
    tnet, rnet = TransitionNet(), RewardNet()
    tnet.load_state_dict(torch.load(os.path.join(RESULTS_DIR, f"transition_net{tag}.pt")))
    rnet.load_state_dict(torch.load(os.path.join(RESULTS_DIR, f"reward_net{tag}.pt")))
    tnet.eval()
    rnet.eval()
    return tnet, rnet


def run_condition(label, seeds, transition_model, reward_model, K, n_steps=N_STEPS):
    """Run one (K, model) condition across seeds; stack the per-seed curves.

    Returns dict: steps, eval_success (n_seeds, n_ckpt), avg_reward (...),
    final_Q (n_seeds, 64, 4).
    """
    evals, rewards, finalQ = [], [], []
    steps = None
    for seed in seeds:
        env = GridWorld(noise_prob=NOISE, seed=seed)
        res = dyna_q(env, transition_model, reward_model, K=K, n_steps=n_steps,
                     record_every=RECORD_EVERY, eval_episodes=EVAL_EPISODES, seed=seed)
        steps = res["steps"]
        evals.append(res["eval_success_rate"])
        rewards.append(res["avg_reward"])
        finalQ.append(res["Q"])
        sr_final = res["eval_success_rate"][-1]
        print(f"  [{label}] seed={seed}: final greedy success={sr_final:.2%}")
    return {
        "label": label,
        "steps": steps,
        "eval_success": np.array(evals),
        "avg_reward": np.array(rewards),
        "final_Q": np.array(finalQ),
    }


def make_training_comparison_gif(checkpoints=(1000, 2000, 3500, 5000, 8000, 12000),
                                 seed=0):
    """Build the side-by-side training-progress GIF (K=0 vs K=10).

    Trains both agents once on the same seed, snapshotting Q at each checkpoint,
    then animates both frozen greedy policies acting in the SAME real env so you
    can watch -- at matched training budgets -- the model-equipped agent reach the
    goal while the baseline is still wandering.
    """
    from plotting import animate_training_comparison

    checkpoints = list(checkpoints)
    n = max(checkpoints)
    tnet, rnet = load_world_model()
    q0 = q_learning(GridWorld(noise_prob=NOISE, seed=seed), n_steps=n,
                    record_every=n, snapshot_steps=checkpoints, seed=seed)
    qK = dyna_q(GridWorld(noise_prob=NOISE, seed=seed), tnet, rnet, K=10, n_steps=n,
                record_every=n, snapshot_steps=checkpoints, seed=seed)
    return animate_training_comparison(GridWorld(noise_prob=NOISE, seed=seed),
                                       q0["snapshots"], qK["snapshots"], checkpoints)


def main(seeds=SEEDS, n_steps=N_STEPS, save=True):
    """Run VI + Q-learning(K=0) + Dyna-Q(K=10), save aggregated curves to npz."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- oracle ceiling: Value Iteration on the true MDP --------------------- #
    env = GridWorld(noise_prob=NOISE, seed=0)
    P, R = env.get_all_transitions()
    V_vi, Q_vi, _ = value_iteration(P, R)
    vi_success, vi_return = evaluate_greedy(Q_vi, n_episodes=200, noise_prob=NOISE)
    print(f"[VI] V(start)={V_vi[0]:.3f}  greedy success={vi_success:.2%}  ret={vi_return:.3f}")

    # --- learned-model agents ----------------------------------------------- #
    tnet, rnet = load_world_model()
    print(f"Running Q-learning (K=0) x{len(seeds)} seeds ...")
    q0 = run_condition("Q-learning (K=0)", seeds, None, None, K=0, n_steps=n_steps)
    print(f"Running Dyna-Q (K=10) x{len(seeds)} seeds ...")
    qK = run_condition("Dyna-Q (K=10)", seeds, tnet, rnet, K=10, n_steps=n_steps)

    # value functions for Figure 4 (mean over seeds; goal stays 0 = absorbing).
    V_q0 = q0["final_Q"].mean(axis=0).max(axis=1)
    V_qK = qK["final_Q"].mean(axis=0).max(axis=1)

    if save:
        np.savez(
            os.path.join(RESULTS_DIR, "p4_curves.npz"),
            steps=q0["steps"],
            q0_eval=q0["eval_success"], qK_eval=qK["eval_success"],
            q0_reward=q0["avg_reward"], qK_reward=qK["avg_reward"],
            vi_success=vi_success, vi_return=vi_return,
            V_vi=V_vi, V_q0=V_q0, V_qK=V_qK,
            seeds=np.array(seeds),
        )
        print(f"Saved P4 curves + value functions to {RESULTS_DIR}/p4_curves.npz")

    return {"q0": q0, "qK": qK, "vi_success": vi_success,
            "V_vi": V_vi, "V_q0": V_q0, "V_qK": V_qK}


if __name__ == "__main__":
    import time
    from plotting import plot_learning_curves, plot_value_heatmaps

    t0 = time.time()
    res = main()

    env = GridWorld(noise_prob=NOISE, seed=0)
    fig3 = plot_learning_curves(
        curves=[
            {"label": "Q-learning (K=0)", "steps": res["q0"]["steps"],
             "data": res["q0"]["eval_success"], "color": "#1f77b4"},
            {"label": "Dyna-Q (K=10)", "steps": res["qK"]["steps"],
             "data": res["qK"]["eval_success"], "color": "#d62728"},
        ],
        vi_ceiling=res["vi_success"],
    )
    print(f"Saved {fig3}")

    fig4 = plot_value_heatmaps(env, [
        ("Value Iteration (V*)", res["V_vi"]),
        ("Q-learning (K=0)", res["V_q0"]),
        ("Dyna-Q (K=10)", res["V_qK"]),
    ])
    print(f"Saved {fig4}")

    gif = make_training_comparison_gif()
    print(f"Saved {gif}")
    print(f"Total P4 time: {time.time() - t0:.1f}s")
