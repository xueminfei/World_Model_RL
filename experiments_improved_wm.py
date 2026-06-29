"""P2 extension: the "improved" world model on the current map.

The baseline world model encodes a state as a content-free 64-dim one-hot. The
improved model instead sees a structured 10-dim observation -- a normalised
absolute-index anchor obs[0] plus a 3x3 CENTERED window of wall/goal markers --
concatenated with the action (14-dim input). One MLP regresses the FULL next
observation obs' (10-dim) AND the reward r.

During training we randomly DROP the anchor obs[0] with probability p, forcing
the net to rely on the window. This script sweeps p in {0, 0.5, 1.0} and, on the
val set, separately tracks the ABSOLUTE part (next-index accuracy) and the
RELATIVE part (per-cell window accuracy), each evaluated with the anchor kept and
with it dropped. The point is to OBSERVE whether the absolute next index becomes
unlearnable once `s` is gone, while the relative window dynamics survive.

Trains on the SAME single map and the SAME 10k random-policy transitions for all
p (a controlled comparison). Saved: results/improved_wm.pt (the p=0.5 weights),
results/improved_wm_metrics.npz, results/figure_improved_wm.png.

NOTE (academic-integrity): this script is tooling. The written interpretation of
the curves is the user's to write.
"""

import os
import time

import numpy as np
import torch

from gridworld import GridWorld
from world_model import (RESULTS_DIR, ImprovedWorldModel, collect_obs_transitions,
                         improved_train_val_split, train_improved_world_model)

torch.set_num_threads(1)

P_VALUES = [0.0, 0.5, 1.0]
N_TRANSITIONS = 10_000
EPOCHS = 20
SEED = 0


def main(p_values=P_VALUES, n_transitions=N_TRANSITIONS, epochs=EPOCHS, seed=SEED,
         save=True):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    env = GridWorld(seed=seed)

    # One dataset, one split, reused for every p -> controlled comparison.
    X, Y_obs, Y_rew, ns_idx = collect_obs_transitions(env, n_transitions, seed=seed)
    train, val = improved_train_val_split(X, Y_obs, Y_rew, ns_idx, seed=seed)
    print(f"Collected {len(X)} transitions (train {len(train['X'])}, "
          f"val {len(val['X'])}); reward +1.0 frac: {(Y_rew > 0).mean():.4f}")

    histories = {}
    main_model = None
    for p in p_values:
        model, hist = train_improved_world_model(train, val, p_drop=p,
                                                 epochs=epochs, seed=seed)
        histories[p] = hist
        f = hist[-1]
        print(f"[p={p}] final val  "
              f"idx_acc: full={f['full']['idx_acc']:.3f} "
              f"state_only={f['state_only']['idx_acc']:.3f} "
              f"win_only={f['win_only']['idx_acc']:.3f}  |  "
              f"win_acc: full={f['full']['win_acc']:.3f} "
              f"state_only={f['state_only']['win_acc']:.3f} "
              f"win_only={f['win_only']['win_acc']:.3f}")
        if p == 0.0:
            main_model = model  # no-drop model: highest same-map accuracy

    if save:
        if main_model is not None:
            torch.save(main_model.state_dict(),
                       os.path.join(RESULTS_DIR, "improved_wm.pt"))
        # Flatten histories into arrays for npz (per p, per group, per metric).
        payload = {"p_values": np.array(p_values), "epochs": epochs}
        for p in p_values:
            for group in ("full", "state_only", "win_only"):
                for metric in ("idx_acc", "win_acc", "rew_mse"):
                    key = f"p{p}_{group}_{metric}"
                    payload[key] = np.array([ep[group][metric]
                                             for ep in histories[p]])
        np.savez(os.path.join(RESULTS_DIR, "improved_wm_metrics.npz"), **payload)
        print(f"Saved improved WM weights + metrics to {RESULTS_DIR}/")

    return histories


if __name__ == "__main__":
    from plotting import plot_improved_wm

    t0 = time.time()
    histories = main()
    fig = plot_improved_wm(histories, P_VALUES)
    print(f"Saved {fig}")
    print(f"Total improved-WM time: {time.time() - t0:.1f}s")
