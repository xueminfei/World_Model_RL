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
from utils import (N_STATES, N_ACTIONS, state_to_index, sa_to_vector,
                   N_WINDOW, IMPROVED_OBS_DIM, IMPROVED_INPUT_DIM,
                   state_to_obs, obs_a_to_vector)

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


# --------------------------------------------------------------------------- #
# 5. Improved world model (P2 extension)
# --------------------------------------------------------------------------- #
# Same map, same random-policy data, but the (s, a) input is the structured
# 77-dim improved encoding (64-dim one-hot anchor + 3x3 centered window(9) +
# action(4)). During training the one-hot anchor block obs[0:64] is randomly
# dropped with prob `p_drop`, forcing the net to read the window. The single net
# predicts the next observation obs': the absolute part as 64 next-state logits
# (classification, like the baseline) and the relative part as the next 3x3
# window (9, regression), plus reward r (1, regression). Validation tracks
# next-state accuracy and per-cell window accuracy, each evaluated twice: with
# the anchor present and with it dropped -- so we can SEE whether the absolute
# part becomes unlearnable once the anchor is removed, while the relative part
# (window) survives.
class ImprovedWorldModel(nn.Module):
    """77 -> 256 -> 256 -> (64 next-state logits, 9 window, 1 reward)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IMPROVED_INPUT_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, N_STATES + N_WINDOW + 1),
        )

    def forward(self, x):
        out = self.net(x)
        logits = out[:, :N_STATES]                       # next-state logits
        window = out[:, N_STATES:N_STATES + N_WINDOW]     # next window
        reward = out[:, N_STATES + N_WINDOW:]             # next reward
        return logits, window, reward


def collect_obs_transitions(env, n_transitions=10_000, seed=0):
    """Random-policy rollout -> improved-encoded transitions.

    Returns X (n,77) input, Y_obs (n,73) target next observation, Y_rew (n,1)
    target reward, and ns_idx (n,) the true next-state index (for accuracy).
    """
    env.rng.seed(seed)
    X = np.zeros((n_transitions, IMPROVED_INPUT_DIM), dtype=np.float32)
    Y_obs = np.zeros((n_transitions, IMPROVED_OBS_DIM), dtype=np.float32)
    Y_rew = np.zeros((n_transitions, 1), dtype=np.float32)
    ns_idx = np.zeros(n_transitions, dtype=np.int64)

    s = env.reset()
    for i in range(n_transitions):
        a = env.rng.randint(0, N_ACTIONS - 1)
        s_next, r, done, _ = env.step(a)
        X[i] = obs_a_to_vector(s, a)
        Y_obs[i] = state_to_obs(s_next)
        Y_rew[i] = r
        ns_idx[i] = state_to_index(s_next)
        s = env.reset() if done else s_next
    return X, Y_obs, Y_rew, ns_idx


def improved_train_val_split(X, Y_obs, Y_rew, ns_idx, val_frac=0.2, seed=0):
    """Shuffle and split improved data into 80% train / 20% val (dict pair)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    perm = rng.permutation(n)
    n_val = int(round(n * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    pack = lambda idx: {
        "X": torch.from_numpy(X[idx]),
        "Y_obs": torch.from_numpy(Y_obs[idx]),
        "Y_rew": torch.from_numpy(Y_rew[idx]),
        "ns_idx": torch.from_numpy(ns_idx[idx]),
    }
    return pack(train_idx), pack(val_idx)


def _drop_blocks(X, drop_state, drop_window):
    """Return a copy of X with the state one-hot and/or window block zeroed."""
    X = X.clone()
    if drop_state:
        X[:, :N_STATES] = 0.0                       # zero the one-hot anchor
    if drop_window:
        X[:, N_STATES:IMPROVED_OBS_DIM] = 0.0        # zero the 3x3 window
    return X


def _improved_metrics(model, data, drop_state=False, drop_window=False):
    """Next-state accuracy + per-cell window accuracy + reward MSE on `data`.

    `drop_state`/`drop_window` zero the corresponding input block before the
    forward pass, so we can isolate each modality's pathway.
    """
    model.eval()
    X = _drop_blocks(data["X"], drop_state, drop_window)
    with torch.no_grad():
        logits, window, reward = model(X)
    # absolute part: argmax over the 64 next-state logits (classification).
    idx_acc = (logits.argmax(dim=1) == data["ns_idx"]).float().mean().item()
    # relative part: threshold each window cell to {-1, 0, +1}, per-cell accuracy.
    win_q = torch.where(window < -0.5, -torch.ones_like(window),
                        torch.where(window > 0.5, torch.ones_like(window),
                                    torch.zeros_like(window)))
    true_win = data["Y_obs"][:, N_STATES:IMPROVED_OBS_DIM]
    win_acc = (win_q == true_win).float().mean().item()
    rew_mse = ((reward - data["Y_rew"]) ** 2).mean().item()
    return {"idx_acc": idx_acc, "win_acc": win_acc, "rew_mse": rew_mse}


# Evaluation conditions: which input modality is available at test time.
_EVAL_CONDS = {
    "full":       dict(drop_state=False, drop_window=False),  # both blocks
    "win_only":   dict(drop_state=True,  drop_window=False),  # window only (no s)
    "state_only": dict(drop_state=False, drop_window=True),   # one-hot s only
}


def train_improved_world_model(train, val, p_drop=0.5, epochs=20, batch_size=128,
                               lr=1e-3, seed=0):
    """Train ImprovedWorldModel, randomly dropping BOTH input blocks.

    Each sample independently drops the state one-hot block (prob `p_drop`) and
    the window block (prob `p_drop`), so the net must learn to predict from
    whichever modality survives -- the absolute pathway (state) and the relative
    pathway (window) both get a learning signal. The absolute target is a 64-way
    classification (CrossEntropy); the window and reward are regressed (MSE).

    Returns (model, history) where history[e] = {cond: metrics} for each of the
    three evaluation conditions in `_EVAL_CONDS`, measured on val after epoch e.
    """
    torch.manual_seed(seed)
    model = ImprovedWorldModel()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    n = len(train["X"])
    win_lo, win_hi = N_STATES, IMPROVED_OBS_DIM

    history = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for b in range(0, n, batch_size):
            idx = perm[b:b + batch_size]
            xb = train["X"][idx].clone()
            if p_drop > 0:  # independent dropout of the state and window blocks
                m = len(idx)
                xb[torch.rand(m) < p_drop, :N_STATES] = 0.0
                xb[torch.rand(m) < p_drop, win_lo:win_hi] = 0.0
            opt.zero_grad()
            logits, window, reward = model(xb)
            loss = (ce(logits, train["ns_idx"][idx])
                    + mse(window, train["Y_obs"][idx][:, win_lo:win_hi])
                    + mse(reward, train["Y_rew"][idx]))
            loss.backward()
            opt.step()
        history.append({
            cond: _improved_metrics(model, val, **kw)
            for cond, kw in _EVAL_CONDS.items()
        })
    return model, history


# --------------------------------------------------------------------------- #
# 6. Adapters so the single ImprovedWorldModel plugs into Dyna-Q
# --------------------------------------------------------------------------- #
# Dyna-Q expects a `transition_model(X) -> logits` and a separate
# `reward_model(X) -> reward`. The improved net returns (logits, window, reward)
# in one call; these thin nn.Module wrappers expose the two pieces (and forward
# `.eval()` correctly) without touching the Dyna-Q loop.
class _ImprovedTransitionAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]  # next-state logits


class _ImprovedRewardAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[2]  # reward (m, 1)


def improved_adapters(model):
    """Return (transition_adapter, reward_adapter) for an ImprovedWorldModel."""
    model.eval()
    return _ImprovedTransitionAdapter(model), _ImprovedRewardAdapter(model)


def load_improved_wm(path=None):
    """Load the frozen ImprovedWorldModel weights (default results/improved_wm.pt)."""
    if path is None:
        path = os.path.join(RESULTS_DIR, "improved_wm.pt")
    model = ImprovedWorldModel()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


if __name__ == "__main__":
    train_world_model()
