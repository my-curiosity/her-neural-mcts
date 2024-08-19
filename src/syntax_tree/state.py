from abc import ABC, abstractmethod
import typing

import numpy as np


class State:
    canonical_state: typing.Any     # s_t
    observation: np.ndarray         # o_t
    action: int     # a_t
    player: int     # player_t
    done: bool      # I(s_t = s_T)
    hash_previous_state: str
    hash : str

    def __init__(self, syntax_tree, observation, done=False, hash=None,
                 production_action=None,
                 previous_state=None,
                 residual_calculated=False):
        self.syntax_tree = syntax_tree
        self.observation = observation
        self.done = done
        self.y_calc = None
        self.hash = hash
        self.production_action = production_action
        self.previous_state = previous_state
        self.residual_calculated=residual_calculated

    def __str__(self):
        return self.hash
