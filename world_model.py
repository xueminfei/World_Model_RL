"""P2: the learned world model (transition net + reward net).

Pipeline:
  1. collect_transitions: random-policy rollouts -> 10000 (s, a, s', r) tuples
     covering the whole state space (NOT just the optimal path).
  2. 80/20 train/val split, encode (s, a) as a 68-dim one-hot vector.
  3. TransitionNet (classification, 64 logits, CrossEntropy, 20 epochs).
     RewardNet (regression, scalar, MSE, 20 epochs).
  4. Validation: transition accuracy should land at ~75-85% -- NOT 100%, because
     the env's 20% noise makes some transitions inherently unpredictable.

The trained nets are frozen and reused by Dyna-Q (P3). Weights and the per-epoch
accuracy history are saved to results/ for plotting Figure 2 and for P3/P4.
"""

import os

import numpy as np
import torch
import torch.nn as nn

# These are tiny MLPs on tiny batches; torch's default intra-op parallelism
# spawns a thread per core and the scheduling overhead dwarfs the compute,
# making it *slower* and pegging every core. One thread is fastest here.
torch.set_num_threads(1)

from gridworld import GridWorld
from utils import N_STATES, N_ACTIONS, state_to_index, sa_to_vector

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
INPUT_DIM = N_STATES + N_ACTIONS  # 68


# --------------------------------------------------------------------------- #
# 1. Data collection
# --------------------------------------------------------------------------- #
def collect_transitions(env, n_transitions=10_000, seed=0):
    """Roll out a random policy and collect `n_transitions` (s, a, s', r) tuples.

    Uses a random policy on purpose so the dataset covers the whole state space,
    including states an optimal agent would never visit. Episodes are reset on
    termination (goal or timeout) and collection continues until the quota.

    Returns
    -------
    X      : (n, 68) float32   -- concat(state one-hot, action one-hot)
    y_next : (n,)   int64      -- index 0..63 of the realized next state
    y_rew  : (n,)   float32    -- realized reward
    """
    env.rng.seed(seed)
    X = np.zeros((n_transitions, INPUT_DIM), dtype=np.float32)
    y_next = np.zeros(n_transitions, dtype=np.int64)
    y_rew = np.zeros(n_transitions, dtype=np.float32)

    s = env.reset()
    for i in range(n_transitions):
        a = env.rng.randint(0, N_ACTIONS - 1)
        s_next, r, done, _ = env.step(a)
        X[i] = sa_to_vector(s, a)
        y_next[i] = state_to_index(s_next)
        y_rew[i] = r
        s = env.reset() if done else s_next
    return X, y_next, y_rew


def train_val_split(X, y_next, y_rew, val_frac=0.2, seed=0):
    """Shuffle and split into 80% train / 20% val. Returns (train, val) dicts."""
    rng = np.random.default_rng(seed)
    n = len(X)
    perm = rng.permutation(n)
    n_val = int(round(n * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    pack = lambda idx: {
        "X": torch.from_numpy(X[idx]),
        "y_next": torch.from_numpy(y_next[idx]),
        "y_rew": torch.from_numpy(y_rew[idx]).unsqueeze(1),
    }
    return pack(train_idx), pack(val_idx)


# --------------------------------------------------------------------------- #
# 2. Networks
# --------------------------------------------------------------------------- #
class TransitionNet(nn.Module):
    """68 -> 256 -> 256 -> 64 logits over the next state (classification)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, N_STATES),
        )

    def forward(self, x):
        return self.net(x)


class RewardNet(nn.Module):
    """68 -> 128 -> 64 -> 1 scalar reward (regression)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# 3. Training
# --------------------------------------------------------------------------- #
def train_transition_model(train, val, epochs=20, batch_size=128, lr=1e-3, seed=0):
    """Train TransitionNet. Returns (model, val_acc_history list of len `epochs`)."""
    torch.manual_seed(seed)
    model = TransitionNet()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n = len(train["X"])

    val_acc_history = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for b in range(0, n, batch_size):
            idx = perm[b:b + batch_size]
            opt.zero_grad()
            logits = model(train["X"][idx])
            loss_fn(logits, train["y_next"][idx]).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(val["X"]).argmax(dim=1)
            acc = (pred == val["y_next"]).float().mean().item()
        val_acc_history.append(acc)
    return model, val_acc_history


def train_reward_model(train, val, epochs=20, batch_size=128, lr=1e-3, seed=0):
    """Train RewardNet. Returns (model, val_mse_history list of len `epochs`)."""
    torch.manual_seed(seed)
    model = RewardNet()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = len(train["X"])

    val_mse_history = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for b in range(0, n, batch_size):
            idx = perm[b:b + batch_size]
            opt.zero_grad()
            pred = model(train["X"][idx])
            loss_fn(pred, train["y_rew"][idx]).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_mse = loss_fn(model(val["X"]), val["y_rew"]).item()
        val_mse_history.append(val_mse)
    return model, val_mse_history


# --------------------------------------------------------------------------- #
# 4. Inspection helper for Figure 2 (top-3 predicted next states)
# --------------------------------------------------------------------------- #
def top_k_next(model, state, action, k=3):
    """Return [(next_index, prob), ...] top-k predictions for a single (s, a)."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(sa_to_vector(state, action)).unsqueeze(0)
        probs = torch.softmax(model(x), dim=1).squeeze(0)
        vals, idx = probs.topk(k)
    return list(zip(idx.tolist(), vals.tolist()))


# --------------------------------------------------------------------------- #
# Train-and-save entry point
# --------------------------------------------------------------------------- #
def train_world_model(n_transitions=10_000, epochs=20, seed=0, save=True, verbose=True):
    """Collect data, train both nets, optionally save frozen weights + metrics.

    Returns a dict with the models, datasets, and histories (used by plotting).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    env = GridWorld(seed=seed)

    X, y_next, y_rew = collect_transitions(env, n_transitions, seed=seed)
    train, val = train_val_split(X, y_next, y_rew, seed=seed)
    if verbose:
        print(f"Collected {len(X)} transitions "
              f"(train {len(train['X'])}, val {len(val['X'])}); "
              f"noisy reward frac +1.0: {(y_rew > 0).mean():.4f}")

    tmodel, acc_hist = train_transition_model(train, val, epochs=epochs, seed=seed)
    rmodel, mse_hist = train_reward_model(train, val, epochs=epochs, seed=seed)
    if verbose:
        print(f"TransitionNet  final val acc : {acc_hist[-1]:.4f}  "
              f"(target 0.75-0.85; not 1.0 because of 20% noise)")
        print(f"RewardNet      final val mse : {mse_hist[-1]:.6f}")

    out = {
        "transition_model": tmodel, "reward_model": rmodel,
        "train": train, "val": val,
        "acc_history": acc_hist, "mse_history": mse_hist,
        "n_transitions": n_transitions, "seed": seed,
    }

    if save:
        tag = f"_n{n_transitions}" if n_transitions != 10_000 else ""
        torch.save(tmodel.state_dict(),
                   os.path.join(RESULTS_DIR, f"transition_net{tag}.pt"))
        torch.save(rmodel.state_dict(),
                   os.path.join(RESULTS_DIR, f"reward_net{tag}.pt"))
        np.savez(os.path.join(RESULTS_DIR, f"world_model_metrics{tag}.npz"),
                 acc_history=np.array(acc_hist),
                 mse_history=np.array(mse_hist),
                 n_transitions=n_transitions, seed=seed)
        if verbose:
            print(f"Saved frozen weights + metrics to {RESULTS_DIR}/ (tag='{tag}')")
    return out


if __name__ == "__main__":
    train_world_model()
