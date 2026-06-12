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
