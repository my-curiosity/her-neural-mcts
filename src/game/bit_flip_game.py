"""
Define the abstract Game class for providing a structure/ interface for agent environments.

"""

import hashlib
import random
import typing

import numpy as np

from src.game.game import GameState, Game


class BitFlipGame(Game):

    def __init__(self, args):
        super().__init__(n_players=1)
        np.seterr(all="raise")
        self.env_name = "Bit_Flip"
        self.args = args
        self.num_bits = args.num_bits
        self.target = np.array(
            [bool(random.getrandbits(1)) for i in range(self.num_bits)]
        )
        self.reader = None

    def getInitialState(self) -> GameState:
        observation = np.array(
            [bool(random.getrandbits(1)) for i in range(self.num_bits)]
        )
        return GameState(None, observation)

    def getDimensions(self) -> typing.Tuple[int, ...]:
        pass

    def getActionSize(self) -> int:
        return self.num_bits

    def getNextState(
        self, state: GameState, action: int, **kwargs
    ) -> typing.Tuple[GameState, float]:
        next_observation = state.observation
        next_observation[action] = not next_observation[action]

        done = np.array_equal(state.observation, self.target)

        next_state = GameState(
            None,
            observation=next_observation,
            done=done,
            production_action=action,
            previous_state=state,
        )

        reward = self.reward(state=next_state)
        next_state.reward = reward
        return next_state, reward

    def reward(self, state):
        return (
            self.args.maximum_reward
            if np.array_equal(state.observation, self.target)
            else self.args.minimum_reward
        )

    def getLegalMoves(self, state: GameState) -> np.ndarray:
        return np.ones(self.num_bits)

    def getGameEnded(self, state: GameState, **kwargs) -> typing.Union[float, int]:
        pass

    def buildObservation(self, state: GameState) -> np.ndarray:
        pass

    def getSymmetries(self, state: GameState, pi: np.ndarray) -> typing.List:
        pass

    def getHash(self, state: GameState) -> typing.Union[str, bytes, int]:
        data = np.ascontiguousarray(state.observation)
        hash1 = hashlib.md5(data).hexdigest()
        string_representation = f"{state.observation}_" f"{hash1}"
        if not isinstance(string_representation, str):
            raise ValueError(
                f"Value to hash is of type {type(string_representation)} should be of type str"
            )
        state.hash = string_representation
        return string_representation
