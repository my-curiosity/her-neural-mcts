import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import random

register(
    id="BitFlip",
    entry_point="src.game.bit_flip_env:BitFlipEnv",
    max_episode_steps=50,
    kwargs={"num_bits": 50, "minimum_reward": -1, "maximum_reward": 0},
)


class BitFlipEnv(gym.Env):
    """
    Simple BitFlip gymnasium env, based on
    https://github.com/NervanaSystems/gym-bit-flip/blob/master/gym_bit_flip/bit_flip.py

    Initializes two different (0,1)^n sequences as initial state and goal. Each step an action i
    is executed, which flips i-th bit in current state. An episode is solved if the goal was reached
    in <= n steps.
    """

    def __init__(self, num_bits, minimum_reward, maximum_reward):
        super(BitFlipEnv, self).__init__()

        self.num_bits = num_bits
        self.minimum_reward = minimum_reward
        self.maximum_reward = maximum_reward

        self.action_space = spaces.Discrete(self.num_bits)
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.MultiBinary(self.num_bits),
                "achieved_goal": spaces.MultiBinary(self.num_bits),
                "desired_goal": spaces.MultiBinary(self.num_bits),
            }
        )

        self.state = np.array([random.getrandbits(1) for _ in range(self.num_bits)])
        self.goal = self.state
        while np.array_equal(self.goal, self.state):
            self.goal = np.array([random.getrandbits(1) for _ in range(self.num_bits)])

    def _terminated(self):
        return np.array_equal(self.state, self.goal)

    def compute_reward(self, state=None, goal=None, info=None):
        state_checked = self.state if state is None else state
        goal_checked = self.goal if goal is None else goal
        return (
            self.maximum_reward
            if np.array_equal(state_checked, goal_checked)
            else self.minimum_reward
        )

    def step(self, action):
        self.state[action] = int(not self.state[action])
        return self._get_obs(), self.compute_reward(), self._terminated(), False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([random.getrandbits(1) for _ in range(self.num_bits)])
        self.goal = options["goal"] if options is not None else self.state
        while np.array_equal(self.goal, self.state):
            self.state = np.array([random.getrandbits(1) for _ in range(self.num_bits)])
        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "observation": np.copy(self.state),
            "achieved_goal": np.copy(self.state),
            "desired_goal": np.copy(self.goal),
        }

    def render(self):
        pass
