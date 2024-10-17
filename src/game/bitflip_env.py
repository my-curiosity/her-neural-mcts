"""
Simple BitFlip gym env, based on https://github.com/NervanaSystems/gym-bit-flip/blob/master/gym_bit_flip/bit_flip.py
"""

import numpy as np
import gym
from gym import spaces
from gym.envs.registration import register
import random

register(
    id="bitflip", entry_point="src.game.bitflip_env:BitFlipEnv", max_episode_steps=100
)


class BitFlipEnv(gym.Env):
    def __init__(self, args):
        super(BitFlipEnv, self).__init__()

        self.num_bits = args.bitflip_num_bits
        self.max_steps = args.bitflip_max_steps

        self.action_space = spaces.Discrete(self.num_bits)
        self.observation_space = spaces.Dict(
            {
                "state": spaces.MultiBinary(self.num_bits),
                "goal": spaces.MultiBinary(self.num_bits),
            }
        )

        self.steps = 0
        self.state = np.array([random.getrandbits(1) for _ in range(self.num_bits)])
        self.goal = np.array([random.getrandbits(1) for _ in range(self.num_bits)])

        self.min_reward = args.minimum_reward
        self.max_reward = args.maximum_reward

    def _terminated(self):
        return np.array_equal(self.state, self.goal) or self.steps >= self.max_steps

    def reward(self, goal=None):
        target = self.goal if goal is None else goal
        return (
            self.max_reward if np.array_equal(self.state, target) else self.min_reward
        )

    def step(self, action):
        self.state[action] = int(not self.state[action])
        self.steps += 1
        return self._get_obs(), self.reward(), self._terminated(), False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.state = np.array([random.getrandbits(1) for _ in range(self.num_bits)])
        self.goal = np.array([random.getrandbits(1) for _ in range(self.num_bits)])
        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "state": self.state,
            "goal": self.goal,
        }

    def render(self):
        pass
