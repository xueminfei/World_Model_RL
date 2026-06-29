"""Encoding utilities shared by the world model and Dyna-Q.

The world model's network input is concat(state one-hot, action one-hot) = 68 dim
(64 states + 4 actions). State/action indexing matches `GridWorld.index`.
"""

import numpy as np

N_STATES = 64
N_ACTIONS = 4
SIZE = 8


def state_to_index(state):
    """(row, col) tuple -> integer index 0..63 (row-major), matching GridWorld."""
    return state[0] * SIZE + state[1]


def index_to_state(index):
    """Integer index 0..63 -> (row, col) tuple."""
    return (index // SIZE, index % SIZE)


def state_to_vector(state):
    """(row, col) -> one-hot vector of length 64 (float32)."""
    vec = np.zeros(N_STATES, dtype=np.float32)
    vec[state_to_index(state)] = 1.0
    return vec


def action_to_vector(action):
    """action int 0..3 -> one-hot vector of length 4 (float32)."""
    vec = np.zeros(N_ACTIONS, dtype=np.float32)
    vec[action] = 1.0
    return vec


def sa_to_vector(state, action):
    """concat(state one-hot, action one-hot) -> 68-dim network input (float32)."""
    return np.concatenate([state_to_vector(state), action_to_vector(action)])


# --------------------------------------------------------------------------- #
# Improved world-model encoding (P2 extension: the "improved WM")
# --------------------------------------------------------------------------- #
# The improved model augments the absolute state with local map structure. The
# observation is a 73-dim vector:
#   obs[0:64]  = 64-dim one-hot of the current cell (the absolute anchor `s`)
#   obs[64:73] = a 3x3 CENTERED window around the agent, cells (r-1..r+1, c-1..c+1)
#                in row-major order. Each cell is marked  -1 (wall OR off-grid =
#                "blocked"),  +1 (the goal),  0 (free). The middle entry is the
#                agent's own cell. Centered (not bottom-right) so walls in ALL
#                four move directions are visible -> the realized displacement is
#                fully predictable from the window.
# During training we randomly DROP the one-hot anchor block obs[0:64] so the
# network is *forced* to read the window. The model predicts the next observation
# obs' -- the absolute part as a 64-way one-hot/softmax over next states, the
# relative part as the next 3x3 window -- plus reward.
N_WINDOW = 9                                       # 3x3 centered window
IMPROVED_OBS_DIM = N_STATES + N_WINDOW             # 73 = one-hot(64) + window(9)
IMPROVED_INPUT_DIM = IMPROVED_OBS_DIM + N_ACTIONS  # 77 = concat(obs, action one-hot)
_WINDOW_OFFSETS = (-1, 0, 1)                       # 3x3 centered window

# Realized 1-step displacements (row, col). Class 0 is "stay" (the passive
# wall/boundary-collision outcome -- NOT a 5th action; the action space stays 4).
DISPLACEMENTS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))  # stay,up,down,left,right
N_DISP = len(DISPLACEMENTS)  # 5
_DISP_TO_CLASS = {d: i for i, d in enumerate(DISPLACEMENTS)}


def _default_map():
    """Default (walls, goal) from GridWorld's class attributes (the training map)."""
    from gridworld import GridWorld  # local import keeps utils import-cycle-free
    return GridWorld.WALLS, GridWorld.GOAL


def _cell_marker(r, c, walls, goal):
    """Marker for one grid cell: -1 blocked (wall/off-grid), +1 goal, 0 free."""
    if not (0 <= r < SIZE and 0 <= c < SIZE):
        return -1.0
    if (r, c) in walls:
        return -1.0
    if (r, c) == goal:
        return 1.0
    return 0.0


def state_to_obs(state, walls=None, goal=None):
    """(row, col) -> 73-dim improved observation (float32). See module note above.

    `walls`/`goal` choose which map the 3x3 window reflects; defaults to the
    training map. Pass a NEW map's walls/goal to encode observations on that map.
    """
    if walls is None or goal is None:
        dwalls, dgoal = _default_map()
        walls = dwalls if walls is None else walls
        goal = dgoal if goal is None else goal
    obs = np.zeros(IMPROVED_OBS_DIM, dtype=np.float32)
    obs[state_to_index(state)] = 1.0  # 64-dim one-hot absolute anchor
    r, c = state
    k = N_STATES
    for dr in _WINDOW_OFFSETS:
        for dc in _WINDOW_OFFSETS:
            obs[k] = _cell_marker(r + dr, c + dc, walls, goal)
            k += 1
    return obs


def obs_a_to_vector(state, action, walls=None, goal=None):
    """concat(73-dim obs, action one-hot) -> 77-dim improved network input."""
    return np.concatenate([state_to_obs(state, walls, goal), action_to_vector(action)])


def build_obs_table(walls=None, goal=None):
    """Return the (64, 73) observation table for a given map (default: training)."""
    return np.stack([state_to_obs(index_to_state(i), walls, goal)
                     for i in range(N_STATES)])


def displacement_class(state, next_state):
    """Realized displacement (state -> next_state) as a class index 0..4."""
    d = (next_state[0] - state[0], next_state[1] - state[1])
    return _DISP_TO_CLASS[d]


def apply_displacement(s_index, disp_class):
    """Absolute next index from a state index + displacement class (s' = s + Delta).

    Off-grid displacements are clamped to "stay" (s' = s), matching the env's
    wall/boundary behaviour. Handy if Dyna-Q later consumes a relative output.
    """
    r, c = index_to_state(s_index)
    dr, dc = DISPLACEMENTS[disp_class]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < SIZE and 0 <= nc < SIZE):
        return s_index
    return state_to_index((nr, nc))


# Precompute the observation for every state once on the training map (fast batch
# encoding for any loop that only holds integer indices).
OBS_TABLE = build_obs_table()


def obs_batch_vectors(s_indices, a_indices, drop_anchor=False, obs_table=None):
    """Build an (m, 77) improved-input batch from state/action index arrays.

    Mirrors the one-hot batch builder but emits the structured obs encoding.
    `obs_table` selects the map (default: training map); pass a NEW map's table
    (from build_obs_table) to encode imagined transitions on that map. If
    `drop_anchor` is True, the one-hot anchor block obs[0:64] is zeroed.
    """
    if obs_table is None:
        obs_table = OBS_TABLE
    s_indices = np.asarray(s_indices)
    a_indices = np.asarray(a_indices)
    m = len(s_indices)
    X = np.zeros((m, IMPROVED_INPUT_DIM), dtype=np.float32)
    X[:, :IMPROVED_OBS_DIM] = obs_table[s_indices]
    if drop_anchor:
        X[:, :N_STATES] = 0.0
    X[np.arange(m), IMPROVED_OBS_DIM + a_indices] = 1.0
    return X
