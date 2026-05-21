"""
Self-contained copy of the parts of `calvin_agent.evaluation.{utils,multistep_sequences}`
that the GR00T eval needs.

This file exists ONLY to avoid importing `calvin_agent.evaluation.utils`, which
unconditionally `import MCIL` at module load — and MCIL requires `pytorch_lightning`.
We don't use MCIL (we use the GR00T policy), so we copy the pure functions here.

Functions exposed:
  - temp_seed
  - get_env_state_for_initial_condition
  - get_sequences

Source (commit at time of copy):
  https://github.com/mees/calvin/blob/main/calvin_models/calvin_agent/evaluation/utils.py
  https://github.com/mees/calvin/blob/main/calvin_models/calvin_agent/evaluation/multistep_sequences.py
"""

import contextlib
import functools
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from itertools import product

import numpy as np
import pyhash
from numpy import pi

hasher = pyhash.fnv1_32()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# from calvin_agent.evaluation.utils
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_env_state_for_initial_condition(initial_condition):
    robot_obs = np.array(
        [
            0.02586889, -0.2313129, 0.5712808, 3.09045411, -0.02908596, 1.50013585, 0.07999963,
            -1.21779124, 1.03987629, 2.11978254, -2.34205014, -0.87015899, 1.64119093, 0.55344928,
            1.0,
        ]
    )
    block_rot_z_range = (pi / 2 - pi / 8, pi / 2 + pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]
    seed = hasher(str(initial_condition.values()))
    with temp_seed(seed):
        np.random.shuffle(block_table)

        scene_obs = np.zeros(24)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = initial_condition["lightbulb"]
        scene_obs[5] = initial_condition["led"]
        if initial_condition["red_block"] == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif initial_condition["red_block"] == "slider_left":
            scene_obs[6:9] = block_slider_left
        else:
            scene_obs[6:9] = block_table[0]
        scene_obs[11] = np.random.uniform(*block_rot_z_range)
        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif initial_condition["red_block"] == "table":
            scene_obs[12:15] = block_table[1]
        else:
            scene_obs[12:15] = block_table[0]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)
        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            scene_obs[18:21] = block_table[1]
        scene_obs[23] = np.random.uniform(*block_rot_z_range)

    return robot_obs, scene_obs


# ---------------------------------------------------------------------------
# from calvin_agent.evaluation.multistep_sequences
# ---------------------------------------------------------------------------
task_categories = {
    "rotate_red_block_right": 1, "rotate_red_block_left": 1,
    "rotate_blue_block_right": 1, "rotate_blue_block_left": 1,
    "rotate_pink_block_right": 1, "rotate_pink_block_left": 1,
    "push_red_block_right": 1, "push_red_block_left": 1,
    "push_blue_block_right": 1, "push_blue_block_left": 1,
    "push_pink_block_right": 1, "push_pink_block_left": 1,
    "move_slider_left": 2, "move_slider_right": 2,
    "open_drawer": 3, "close_drawer": 3,
    "lift_red_block_table": 4, "lift_red_block_slider": 5, "lift_red_block_drawer": 6,
    "lift_blue_block_table": 4, "lift_blue_block_slider": 5, "lift_blue_block_drawer": 6,
    "lift_pink_block_table": 4, "lift_pink_block_slider": 5, "lift_pink_block_drawer": 6,
    "place_in_slider": 7, "place_in_drawer": 7,
    "turn_on_lightbulb": 8, "turn_off_lightbulb": 8,
    "turn_on_led": 8, "turn_off_led": 8,
    "push_into_drawer": 9,
    "stack_block": 10, "unstack_block": 11,
}

tasks = {
    "rotate_red_block_right":  [{"condition": {"red_block": "table", "grasped": 0},  "effect": {"red_block": "table"}}],
    "rotate_red_block_left":   [{"condition": {"red_block": "table", "grasped": 0},  "effect": {"red_block": "table"}}],
    "rotate_blue_block_right": [{"condition": {"blue_block": "table", "grasped": 0}, "effect": {"blue_block": "table"}}],
    "rotate_blue_block_left":  [{"condition": {"blue_block": "table", "grasped": 0}, "effect": {"blue_block": "table"}}],
    "rotate_pink_block_right": [{"condition": {"pink_block": "table", "grasped": 0}, "effect": {"pink_block": "table"}}],
    "rotate_pink_block_left":  [{"condition": {"pink_block": "table", "grasped": 0}, "effect": {"pink_block": "table"}}],
    "push_red_block_right":    [{"condition": {"red_block": "table", "grasped": 0},  "effect": {"red_block": "table"}}],
    "push_red_block_left":     [{"condition": {"red_block": "table", "grasped": 0},  "effect": {"red_block": "table"}}],
    "push_blue_block_right":   [{"condition": {"blue_block": "table", "grasped": 0}, "effect": {"blue_block": "table"}}],
    "push_blue_block_left":    [{"condition": {"blue_block": "table", "grasped": 0}, "effect": {"blue_block": "table"}}],
    "push_pink_block_right":   [{"condition": {"pink_block": "table", "grasped": 0}, "effect": {"pink_block": "table"}}],
    "push_pink_block_left":    [{"condition": {"pink_block": "table", "grasped": 0}, "effect": {"pink_block": "table"}}],
    "move_slider_left":  [{"condition": {"slider": "right", "grasped": 0}, "effect": {"slider": "left"}}],
    "move_slider_right": [{"condition": {"slider": "left",  "grasped": 0}, "effect": {"slider": "right"}}],
    "open_drawer":  [{"condition": {"drawer": "closed", "grasped": 0}, "effect": {"drawer": "open"}}],
    "close_drawer": [{"condition": {"drawer": "open",   "grasped": 0}, "effect": {"drawer": "closed"}}],
    "lift_red_block_table":  [{"condition": {"red_block": "table",  "grasped": 0}, "effect": {"red_block": "grasped", "grasped": 1}}],
    "lift_red_block_slider": [
        {"condition": {"red_block": "slider_left",  "slider": "right", "grasped": 0}, "effect": {"red_block": "grasped", "grasped": 1}},
        {"condition": {"red_block": "slider_right", "slider": "left",  "grasped": 0}, "effect": {"red_block": "grasped", "grasped": 1}},
    ],
    "lift_red_block_drawer": [{"condition": {"red_block": "drawer", "drawer": "open", "grasped": 0}, "effect": {"red_block": "grasped", "grasped": 1}}],
    "lift_blue_block_table": [{"condition": {"blue_block": "table", "grasped": 0}, "effect": {"blue_block": "grasped", "grasped": 1}}],
    "lift_blue_block_slider": [
        {"condition": {"blue_block": "slider_left",  "slider": "right", "grasped": 0}, "effect": {"blue_block": "grasped", "grasped": 1}},
        {"condition": {"blue_block": "slider_right", "slider": "left",  "grasped": 0}, "effect": {"blue_block": "grasped", "grasped": 1}},
    ],
    "lift_blue_block_drawer": [{"condition": {"blue_block": "drawer", "drawer": "open", "grasped": 0}, "effect": {"blue_block": "grasped", "grasped": 1}}],
    "lift_pink_block_table": [{"condition": {"pink_block": "table", "grasped": 0}, "effect": {"pink_block": "grasped", "grasped": 1}}],
    "lift_pink_block_slider": [
        {"condition": {"pink_block": "slider_left",  "slider": "right", "grasped": 0}, "effect": {"pink_block": "grasped", "grasped": 1}},
        {"condition": {"pink_block": "slider_right", "slider": "left",  "grasped": 0}, "effect": {"pink_block": "grasped", "grasped": 1}},
    ],
    "lift_pink_block_drawer": [{"condition": {"pink_block": "drawer", "drawer": "open", "grasped": 0}, "effect": {"pink_block": "grasped", "grasped": 1}}],
    "place_in_slider": [
        {"condition": {"red_block":  "grasped", "slider": "right", "grasped": 1}, "effect": {"red_block":  "slider_right", "grasped": 0}},
        {"condition": {"red_block":  "grasped", "slider": "left",  "grasped": 1}, "effect": {"red_block":  "slider_left",  "grasped": 0}},
        {"condition": {"blue_block": "grasped", "slider": "right", "grasped": 1}, "effect": {"blue_block": "slider_right", "grasped": 0}},
        {"condition": {"blue_block": "grasped", "slider": "left",  "grasped": 1}, "effect": {"blue_block": "slider_left",  "grasped": 0}},
        {"condition": {"pink_block": "grasped", "slider": "right", "grasped": 1}, "effect": {"pink_block": "slider_right", "grasped": 0}},
        {"condition": {"pink_block": "grasped", "slider": "left",  "grasped": 1}, "effect": {"pink_block": "slider_left",  "grasped": 0}},
    ],
    "place_in_drawer": [
        {"condition": {"red_block":  "grasped", "drawer": "open", "grasped": 1}, "effect": {"red_block":  "drawer", "grasped": 0}},
        {"condition": {"blue_block": "grasped", "drawer": "open", "grasped": 1}, "effect": {"blue_block": "drawer", "grasped": 0}},
        {"condition": {"pink_block": "grasped", "drawer": "open", "grasped": 1}, "effect": {"pink_block": "drawer", "grasped": 0}},
    ],
    "stack_block": [
        {"condition": {"red_block":  "grasped", "blue_block": "table", "grasped": 1}, "effect": {"red_block": "stacked_top",  "blue_block": "stacked_bottom", "grasped": 0}},
        {"condition": {"red_block":  "grasped", "pink_block": "table", "grasped": 1}, "effect": {"red_block": "stacked_top",  "pink_block": "stacked_bottom", "grasped": 0}},
        {"condition": {"blue_block": "grasped", "red_block":  "table", "grasped": 1}, "effect": {"blue_block": "stacked_top", "red_block":  "stacked_bottom", "grasped": 0}},
        {"condition": {"blue_block": "grasped", "pink_block": "table", "grasped": 1}, "effect": {"blue_block": "stacked_top", "pink_block": "stacked_bottom", "grasped": 0}},
        {"condition": {"pink_block": "grasped", "red_block":  "table", "grasped": 1}, "effect": {"pink_block": "stacked_top", "red_block":  "stacked_bottom", "grasped": 0}},
        {"condition": {"pink_block": "grasped", "blue_block": "table", "grasped": 1}, "effect": {"pink_block": "stacked_top", "blue_block": "stacked_bottom", "grasped": 0}},
    ],
    "unstack_block": [
        {"condition": {"red_block":  "stacked_top", "blue_block": "stacked_bottom", "grasped": 0}, "effect": {"red_block":  "table", "blue_block": "table"}},
        {"condition": {"red_block":  "stacked_top", "pink_block": "stacked_bottom", "grasped": 0}, "effect": {"red_block":  "table", "pink_block": "table"}},
        {"condition": {"blue_block": "stacked_top", "red_block":  "stacked_bottom", "grasped": 0}, "effect": {"blue_block": "table", "red_block":  "table"}},
        {"condition": {"blue_block": "stacked_top", "pink_block": "stacked_bottom", "grasped": 0}, "effect": {"blue_block": "table", "pink_block": "table"}},
        {"condition": {"pink_block": "stacked_top", "red_block":  "stacked_bottom", "grasped": 0}, "effect": {"pink_block": "table", "red_block":  "table"}},
        {"condition": {"pink_block": "stacked_top", "blue_block": "stacked_bottom", "grasped": 0}, "effect": {"pink_block": "table", "blue_block": "table"}},
    ],
    "turn_on_lightbulb":  [{"condition": {"lightbulb": 0, "grasped": 0}, "effect": {"lightbulb": 1}}],
    "turn_off_lightbulb": [{"condition": {"lightbulb": 1, "grasped": 0}, "effect": {"lightbulb": 0}}],
    "turn_on_led":        [{"condition": {"led": 0, "grasped": 0}, "effect": {"led": 1}}],
    "turn_off_led":       [{"condition": {"led": 1, "grasped": 0}, "effect": {"led": 0}}],
    "push_into_drawer": [
        {"condition": {"red_block":  "table", "blue_block": ["slider_right", "slider_left"], "pink_block": ["slider_right", "slider_left"], "drawer": "open", "grasped": 0}, "effect": {"red_block":  "drawer", "grasped": 0}},
        {"condition": {"blue_block": "table", "red_block":  ["slider_right", "slider_left"], "pink_block": ["slider_right", "slider_left"], "drawer": "open", "grasped": 0}, "effect": {"blue_block": "drawer", "grasped": 0}},
        {"condition": {"pink_block": "table", "blue_block": ["slider_right", "slider_left"], "red_block":  ["slider_right", "slider_left"], "drawer": "open", "grasped": 0}, "effect": {"pink_block": "drawer", "grasped": 0}},
    ],
}


def _check_condition(state, condition):
    for k, v in condition.items():
        if isinstance(v, (str, int)):
            if not state[k] == v:
                return False
        elif isinstance(v, list):
            if not state[k] in v:
                return False
        else:
            raise TypeError
    return True


def _update_state(state, effect):
    next_state = deepcopy(state)
    for k, v in effect.items():
        next_state[k] = v
    return next_state


def _valid_task(curr_state, task):
    next_states = []
    for _task in task:
        if _check_condition(curr_state, _task["condition"]):
            next_states.append(_update_state(curr_state, _task["effect"]))
    return next_states


def _check_sequence(state, seq):
    for task_name in seq:
        states = _valid_task(state, tasks[task_name])
        if len(states) != 1:
            return False
        state = states[0]
    categories = [task_categories[name] for name in seq]
    return len(categories) == len(set(categories))


def _get_sequences_for_state2(args):
    state, num_sequences, i = args
    np.random.seed(i)
    seq_len = 5
    results = []
    while len(results) < num_sequences:
        seq = np.random.choice(list(tasks.keys()), size=seq_len, replace=False)
        if _check_sequence(state, seq):
            results.append(seq)
    return results


def _flatten(t):
    return [tuple(item.tolist()) for sublist in t for item in sublist]


@functools.lru_cache
def get_sequences(num_sequences: int = 1000, num_workers=None):
    possible_conditions = {
        "led": [0, 1], "lightbulb": [0, 1],
        "slider": ["right", "left"], "drawer": ["closed", "open"],
        "red_block":  ["table", "slider_right", "slider_left"],
        "blue_block": ["table", "slider_right", "slider_left"],
        "pink_block": ["table", "slider_right", "slider_left"],
        "grasped": [0],
    }
    f = lambda l: l.count("table") in [1, 2] and l.count("slider_right") < 2 and l.count("slider_left") < 2
    value_combinations = filter(f, product(*possible_conditions.values()))
    initial_states = [dict(zip(possible_conditions.keys(), vals)) for vals in value_combinations]

    num_sequences_per_state = list(map(len, np.array_split(range(num_sequences), len(initial_states))))
    logger.info("Generating evaluation sequences.")
    with temp_seed(0):
        num_workers = multiprocessing.cpu_count() if num_workers is None else num_workers
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = _flatten(
                executor.map(
                    _get_sequences_for_state2,
                    zip(initial_states, num_sequences_per_state, range(len(initial_states))),
                )
            )
        results = list(zip(np.repeat(initial_states, num_sequences_per_state), results))
        np.random.shuffle(results)
    return results
