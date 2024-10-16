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
        self.observation_space = spaces.Box(0, 1, shape=(self.num_bits,), dtype=int)

        self.steps = 0
        self.state = np.array([random.getrandbits(1) for _ in range(self.num_bits)])
        self.goal = np.array([random.getrandbits(1) for _ in range(self.num_bits)])

        self.min_reward = args.minimum_reward
        self.max_reward = args.maximum_reward

    def _terminated(self):
        return (self.state == self.goal).all() or self.steps >= self.max_steps

    def _reward(self):
        return self.max_reward if (self.state == self.goal).all() else self.min_reward

    def step(self, action):
        self.state[action] = int(not self.state[action])
        self.steps += 1
        return self._get_obs(), self._reward(), self._terminated(), False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.state = np.array([random.getrandbits(1) for _ in range(self.num_bits)])
        self.goal = np.array([random.getrandbits(1) for _ in range(self.num_bits)])
        return self._get_obs(), {}

    def _get_obs(self):
        return self.state

    def render(self):
        pass
