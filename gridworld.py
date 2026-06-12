"""P1: The stochastic 8x8 gridworld environment.

State is an integer tuple (row, col) with row, col in {0..7} -> 64 states.
Start is top-left (0, 0); goal is bottom-right (7, 7). Movement is noisy: with
probability `noise_prob` the chosen action is ignored and a uniformly random
direction is taken instead. Moving into a wall or off the grid keeps the agent
in place.
"""

import random as _random
from collections import deque

import numpy as np


class GridWorld:
    SIZE = 8
    START = (0, 0)
    GOAL = (7, 7)

    # 10 fixed wall cells; a path from start to goal is guaranteed (asserted).
    WALLS = frozenset({
        (1, 1), (1, 2), (1, 3),
        (2, 6),
        (3, 3), (3, 4), (3, 5),
        (4, 6),
        (5, 1), (5, 2),
    })

    # Actions: 0=up, 1=down, 2=left, 3=right.
    N_ACTIONS = 4
    _DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    MAX_STEPS = 100
    GOAL_REWARD = 1.0
    STEP_REWARD = -0.01

    def __init__(self, noise_prob=0.2, seed=None):
        self.noise_prob = noise_prob
        self.n_states = self.SIZE * self.SIZE
        self.rng = _random.Random(seed)
        assert self._path_exists(), "No path from start to goal — fix WALLS."
        self.state = None
        self.t = 0

    # --- index helpers -----------------------------------------------------
    @staticmethod
    def index(state):
        """(row, col) -> integer index 0..63."""
        return state[0] * GridWorld.SIZE + state[1]

    @staticmethod
    def to_state(index):
        """integer index 0..63 -> (row, col)."""
        return (index // GridWorld.SIZE, index % GridWorld.SIZE)

    # --- geometry ----------------------------------------------------------
    def _in_bounds(self, r, c):
        return 0 <= r < self.SIZE and 0 <= c < self.SIZE

    def _is_wall(self, r, c):
        return (r, c) in self.WALLS

    def _move(self, state, direction):
        """Deterministic move in `direction`; stay put if blocked."""
        r, c = state
        dr, dc = self._DELTAS[direction]
        nr, nc = r + dr, c + dc
        if not self._in_bounds(nr, nc) or self._is_wall(nr, nc):
            return (r, c)
        return (nr, nc)

    def _path_exists(self):
        """BFS over free cells to confirm start can reach goal."""
        seen = {self.START}
        queue = deque([self.START])
        while queue:
            cur = queue.popleft()
            if cur == self.GOAL:
                return True
            for d in range(self.N_ACTIONS):
                nxt = self._move(cur, d)
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return False

    # --- core API ----------------------------------------------------------
    def reset(self):
        """Place the agent at the start and return its state (row, col)."""
        self.state = self.START
        self.t = 0
        return self.state

    def step(self, action):
        """Apply `action` with noise.

        Returns (next_state, reward, done, info) where
        info = {'true_next': (r, c), 'noisy': bool}. `true_next` is where the
        intended action would have led with no noise; `noisy` is whether the
        noise branch fired this step.
        """
        assert self.state is not None, "call reset() before step()."
        true_next = self._move(self.state, action)

        noisy = self.rng.random() < self.noise_prob
        direction = self.rng.randint(0, self.N_ACTIONS - 1) if noisy else action
        next_state = self._move(self.state, direction)

        self.t += 1
        if next_state == self.GOAL:
            reward, done = self.GOAL_REWARD, True
        else:
            reward, done = self.STEP_REWARD, self.t >= self.MAX_STEPS

        self.state = next_state
        info = {'true_next': true_next, 'noisy': noisy}
        return next_state, reward, done, info

    def get_all_transitions(self):
        """Return the true transition tensor P[s, a, s'] and reward R[s, a].

        Used only by Value Iteration (the oracle baseline). The goal is an
        absorbing state with zero reward so its value is 0 under Value Iteration.
        """
        n, n_a = self.n_states, self.N_ACTIONS
        P = np.zeros((n, n_a, n), dtype=np.float64)
        R = np.zeros((n, n_a), dtype=np.float64)

        p_other = self.noise_prob / n_a
        p_intended = (1.0 - self.noise_prob) + p_other

        for r in range(self.SIZE):
            for c in range(self.SIZE):
                s = self.index((r, c))
                if (r, c) == self.GOAL:
                    P[s, :, s] = 1.0  # absorbing, zero reward
                    continue
                for a in range(n_a):
                    for d in range(n_a):
                        prob = p_intended if d == a else p_other
                        nxt = self._move((r, c), d)
                        ns = self.index(nxt)
                        P[s, a, ns] += prob
                        rew = self.GOAL_REWARD if nxt == self.GOAL else self.STEP_REWARD
                        R[s, a] += prob * rew
        return P, R


if __name__ == "__main__":
    # Quick sanity check: a few random rollouts and transition-tensor checks.
    env = GridWorld(seed=0)
    P, R = env.get_all_transitions()
    assert np.allclose(P.sum(axis=2), 1.0), "transition rows must sum to 1"
    print(f"States: {env.n_states}, actions: {env.N_ACTIONS}, walls: {len(env.WALLS)}")
    print(f"P shape: {P.shape}, R shape: {R.shape}, path exists: {env._path_exists()}")

    reached = 0
    for _ in range(100):
        s = env.reset()
        for _ in range(env.MAX_STEPS):
            s, rew, done, info = env.step(env.rng.randint(0, 3))
            if done:
                reached += rew > 0
                break
    print(f"Random-policy goal reach rate over 100 episodes: {reached}/100")
