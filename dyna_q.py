"""P3: Dyna-Q, plain Q-learning, and Value Iteration.

Three learners that share the same Bellman update but differ in where the
experience comes from:

  * value_iteration : the ORACLE ceiling. Plans with the *true* transition
    tensor P[s,a,s'] from `env.get_all_transitions()`; no learning, no samples.
  * dyna_q(..., K>0) : after each REAL environment step (one ordinary
    Q-learning update) it draws K *imagined* steps from the frozen world model
    and applies the same Q update to them -> more learning per real step.
  * q_learning = dyna_q(..., K=0) : the no-model baseline.

Three design choices from the project spec, all enforced below:
  1. Imagined (s, a) pairs are sampled ONLY from pairs the agent has actually
     visited (the replay buffer) -- never from unseen states, where the model's
     prediction would be unreliable.
  2. The world model is trained and FROZEN before RL starts; its weights are
     never touched in this file (eval mode, no_grad) so we measure the model's
     pure effect.
  3. Baseline = Dyna-Q(K=0); ceiling = Value Iteration.

The transition net outputs logits over the 64 next states; an imagined next
state is *sampled* from that softmax (not argmax) so the imagined rollouts carry
the model's learned stochasticity. The reward net regresses the scalar reward.
A predicted next state equal to the goal is treated as terminal (no bootstrap).
"""

import numpy as np
import torch

from gridworld import GridWorld
from utils import N_STATES, N_ACTIONS, state_to_index, index_to_state

# Same rationale as world_model.py: single-thread is fastest for these tiny MLPs
# queried on small batches, and avoids pegging every core.
torch.set_num_threads(1)

GOAL_INDEX = state_to_index(GridWorld.GOAL)  # 63


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _greedy_action(Q, s_idx, rng):
    """argmax over actions at state `s_idx`, breaking ties uniformly at random."""
    row = Q[s_idx]
    best = np.flatnonzero(row == row.max())
    return int(best[rng.integers(len(best))]) if len(best) > 1 else int(best[0])


def _sa_batch_vectors(s_indices, a_indices):
    """Build an (m, 68) one-hot batch concat(state, action) without Python loops."""
    m = len(s_indices)
    X = np.zeros((m, N_STATES + N_ACTIONS), dtype=np.float32)
    rows = np.arange(m)
    X[rows, s_indices] = 1.0
    X[rows, N_STATES + a_indices] = 1.0
    return X


def evaluate_greedy(Q, n_episodes=200, noise_prob=0.2, max_steps=None, seed=12345,
                    walls=None, goal=None, start=None):
    """Run the greedy policy of `Q` and return (success_rate, mean_return).

    A fresh env (its own RNG) is used so evaluation never perturbs training.
    `success` = reached the goal before timeout. `walls`/`goal`/`start` pick the
    map to evaluate on (default: training map) -- needed so a new-map run is
    scored on the new map, not the training one.
    """
    env = GridWorld(noise_prob=noise_prob, seed=seed, walls=walls, goal=goal, start=start)
    if max_steps is None:
        max_steps = env.MAX_STEPS
    rng = np.random.default_rng(seed)
    successes, returns = 0, []
    for _ in range(n_episodes):
        s = env.reset()
        total = 0.0
        for _ in range(max_steps):
            a = _greedy_action(Q, state_to_index(s), rng)
            s, r, done, _ = env.step(a)
            total += r
            if done:
                successes += r > 0
                break
        returns.append(total)
    return successes / n_episodes, float(np.mean(returns))


# --------------------------------------------------------------------------- #
# Dyna-Q (and Q-learning as the K=0 special case)
# --------------------------------------------------------------------------- #
def dyna_q(env, transition_model, reward_model, K=10, alpha=0.1, gamma=0.95,
           n_steps=50_000, epsilon_start=1.0, epsilon_end=0.05,
           record_every=500, eval_episodes=0, snapshot_steps=None, seed=0,
           imagine="neural", sa_encoder=None):
    """Train a tabular agent for `n_steps` REAL env steps with K imagined updates.

    Each real step does one standard Q-learning update, then (if K>0) K extra
    updates whose (next state, reward) come from the chosen *imagination source*.
    This isolates two orthogonal knobs: the imagination loop (K) and where its
    synthetic transitions come from (`imagine`).

    Parameters
    ----------
    transition_model, reward_model : frozen nn.Module or None
        The learned world model. Only used when imagine="neural".
    imagine : {"neural", "tabular", "replay"}
        Source of the K imagined transitions (ignored when K=0):
          * "neural"  : sample (s,a) from visited pairs; the frozen MLP world
                        model predicts (s', r). The project's Dyna-Q.
          * "tabular" : classic Sutton Dyna-Q. A lookup table remembers the LAST
                        observed (s', r) for each visited (s,a) and replays it --
                        imagination WITHOUT a learned/neural model.
          * "replay"  : experience replay. Re-apply the Q update to K real,
                        previously observed transitions (s,a,r,s') drawn from a
                        buffer -- no model at all, only real data reused.
    record_every : log recent-20-episode avg reward & success rate this often.
    eval_episodes : if >0, also log greedy-policy success rate over this many
        fresh episodes at each checkpoint (cleaner Figure 3 curve).
    snapshot_steps : optional iterable of real-step counts at which to stash a
        copy of the current Q table (for the training-progress comparison GIF).

    Returns a dict: Q table, the recorded learning-curve arrays, K, seed; plus
    a "snapshots" {step: Q.copy()} dict if snapshot_steps was given.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    use_neural = (K > 0 and imagine == "neural"
                  and transition_model is not None and reward_model is not None)
    # How to encode imagined (s, a) pairs into the world model's input. Defaults
    # to the 68-dim one-hot builder; the improved WM passes obs_batch_vectors
    # (77-dim structured obs) so the same imagination loop works unchanged.
    encode_sa = sa_encoder if sa_encoder is not None else _sa_batch_vectors
    if use_neural:
        transition_model.eval()
        reward_model.eval()

    Q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)

    # Set of *visited* (s, a) pairs (deduped) -- the only seeds neural/tabular
    # imagination is ever allowed to start from (never an unseen pair).
    visited_set = set()
    visited_list = []
    # "tabular" model: last observed (s', r, done) per visited (s, a).
    model_table = {}
    # "replay" buffer: every real transition (s, a, r, s', done) seen so far.
    replay = []

    eps_decay = (epsilon_start - epsilon_end)

    # Rolling window over the last 20 finished episodes.
    from collections import deque
    recent_returns = deque(maxlen=20)
    recent_success = deque(maxlen=20)

    rec_steps, rec_avg_reward, rec_success = [], [], []
    rec_eval_success = []
    snapshot_set = set(snapshot_steps) if snapshot_steps is not None else set()
    snapshots = {}

    def checkpoint(step):
        rec_steps.append(step)
        rec_avg_reward.append(np.mean(recent_returns) if recent_returns else np.nan)
        rec_success.append(np.mean(recent_success) if recent_success else np.nan)
        if eval_episodes > 0:
            # Fixed eval seed across checkpoints: the same eval episodes every
            # time, so the curve reflects policy improvement, not eval-set noise.
            sr, _ = evaluate_greedy(Q, n_episodes=eval_episodes,
                                    noise_prob=env.noise_prob, seed=99,
                                    walls=env.WALLS, goal=env.GOAL, start=env.START)
            rec_eval_success.append(sr)

    s = env.reset()
    ep_return, ep_len = 0.0, 0

    for step in range(1, n_steps + 1):
        epsilon = max(epsilon_end, epsilon_start - eps_decay * step / n_steps)
        s_idx = state_to_index(s)

        # epsilon-greedy action in the REAL env.
        if rng.random() < epsilon:
            a = int(rng.integers(N_ACTIONS))
        else:
            a = _greedy_action(Q, s_idx, rng)

        s_next, r, done, _ = env.step(a)
        ns_idx = state_to_index(s_next)
        ep_return += r
        ep_len += 1

        # --- 1 real Q-learning update -------------------------------------- #
        target = r if done else r + gamma * Q[ns_idx].max()
        Q[s_idx, a] += alpha * (target - Q[s_idx, a])

        # Remember this (s, a) as a legal seed for imagination, and record the
        # observed transition for the tabular model / replay buffer.
        if (s_idx, a) not in visited_set:
            visited_set.add((s_idx, a))
            visited_list.append((s_idx, a))
        model_table[(s_idx, a)] = (ns_idx, r, done)
        replay.append((s_idx, a, r, ns_idx, done))

        # --- K imagined updates from the chosen imagination source --------- #
        if K > 0:
            if use_neural and visited_list:
                pick = rng.integers(len(visited_list), size=K)
                sa = np.asarray(visited_list, dtype=np.int64)[pick]
                im_s, im_a = sa[:, 0], sa[:, 1]
                X = torch.from_numpy(encode_sa(im_s, im_a))
                with torch.no_grad():
                    probs = torch.softmax(transition_model(X), dim=1)
                    im_ns = torch.multinomial(probs, 1).squeeze(1).numpy()
                    im_r = reward_model(X).squeeze(1).numpy()
                im_terminal = im_ns == GOAL_INDEX
                im_target = im_r + gamma * Q[im_ns].max(axis=1) * (~im_terminal)
                for j in range(K):  # K is tiny (e.g. 10), a small loop is fine.
                    si, ai = int(im_s[j]), int(im_a[j])
                    Q[si, ai] += alpha * (im_target[j] - Q[si, ai])

            elif imagine == "tabular" and visited_list:
                # Classic Dyna-Q: replay the memorised (s', r) for sampled pairs.
                pick = rng.integers(len(visited_list), size=K)
                for j in pick:
                    si, ai = visited_list[j]
                    ns, rr, dn = model_table[(si, ai)]
                    tgt = rr if dn else rr + gamma * Q[ns].max()
                    Q[si, ai] += alpha * (tgt - Q[si, ai])

            elif imagine == "replay" and replay:
                # Experience replay: re-apply the Q update to real transitions.
                pick = rng.integers(len(replay), size=K)
                for j in pick:
                    si, ai, rr, ns, dn = replay[j]
                    tgt = rr if dn else rr + gamma * Q[ns].max()
                    Q[si, ai] += alpha * (tgt - Q[si, ai])

        # --- episode bookkeeping ------------------------------------------- #
        if done:
            recent_returns.append(ep_return)
            recent_success.append(1.0 if r > 0 else 0.0)
            s = env.reset()
            ep_return, ep_len = 0.0, 0
        else:
            s = s_next

        if step % record_every == 0:
            checkpoint(step)
        if step in snapshot_set:
            snapshots[step] = Q.copy()

    out = {
        "Q": Q,
        "steps": np.array(rec_steps),
        "avg_reward": np.array(rec_avg_reward),
        "success_rate": np.array(rec_success),
        "K": K,
        "imagine": imagine if K > 0 else "none",
        "seed": seed,
    }
    if eval_episodes > 0:
        out["eval_success_rate"] = np.array(rec_eval_success)
    if snapshot_steps is not None:
        out["snapshots"] = snapshots
    return out


def q_learning(env, alpha=0.1, gamma=0.95, n_steps=50_000,
               epsilon_start=1.0, epsilon_end=0.05, record_every=500,
               eval_episodes=0, snapshot_steps=None, seed=0):
    """Plain Q-learning = Dyna-Q with no world model (K=0). The baseline."""
    return dyna_q(env, None, None, K=0, alpha=alpha, gamma=gamma,
                  n_steps=n_steps, epsilon_start=epsilon_start,
                  epsilon_end=epsilon_end, record_every=record_every,
                  eval_episodes=eval_episodes, snapshot_steps=snapshot_steps, seed=seed)


# --------------------------------------------------------------------------- #
# Value Iteration (oracle ceiling, uses the TRUE transition tensor)
# --------------------------------------------------------------------------- #
def value_iteration(P, R, gamma=0.95, threshold=1e-6):
    """Solve the MDP exactly from the true (P, R). Returns (V, Q, policy).

    P : (S, A, S') true transition probabilities; R : (S, A) expected reward.
    Converges in well under a second for 64 states. V is the Figure-4 colour
    truth; its greedy policy is the Figure-3 ceiling.
    """
    n_states, n_actions, _ = P.shape
    V = np.zeros(n_states, dtype=np.float64)
    while True:
        Q = R + gamma * P.dot(V)            # (S, A)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < threshold:
            V = V_new
            break
        V = V_new
    Q = R + gamma * P.dot(V)
    policy = Q.argmax(axis=1)
    return V, Q, policy


# --------------------------------------------------------------------------- #
# Self-check
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import os
    from world_model import TransitionNet, RewardNet, RESULTS_DIR

    env = GridWorld(seed=0)

    # Value Iteration sanity: optimal greedy policy should solve the env well.
    P, R = env.get_all_transitions()
    V, Qstar, pi = value_iteration(P, R)
    vi_sr, vi_ret = evaluate_greedy(Qstar, n_episodes=200, noise_prob=env.noise_prob)
    print(f"[VI]  V(start)={V[0]:.3f}  greedy success={vi_sr:.2%}  ret={vi_ret:.3f}")

    # Load the frozen world model if it exists, then run a short Dyna-Q vs Q-learning.
    tpath = os.path.join(RESULTS_DIR, "transition_net.pt")
    rpath = os.path.join(RESULTS_DIR, "reward_net.pt")
    if os.path.exists(tpath) and os.path.exists(rpath):
        tnet, rnet = TransitionNet(), RewardNet()
        tnet.load_state_dict(torch.load(tpath))
        rnet.load_state_dict(torch.load(rpath))

        n = 20_000
        q0 = q_learning(GridWorld(seed=1), n_steps=n, seed=1)
        qK = dyna_q(GridWorld(seed=1), tnet, rnet, K=10, n_steps=n, seed=1)
        sr0, _ = evaluate_greedy(q0["Q"], noise_prob=0.2)
        srK, _ = evaluate_greedy(qK["Q"], noise_prob=0.2)
        print(f"[{n} steps]  Q-learning greedy success={sr0:.2%}  "
              f"|  Dyna-Q(K=10) greedy success={srK:.2%}")
    else:
        print("(world model weights not found; run world_model.py first to test Dyna-Q)")
