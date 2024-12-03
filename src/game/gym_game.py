import copy
import random
import typing
import warnings

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from pcfg import PCFG
from src.utils.get_grammar import add_prior
from src.game.game import Game, GameState
from src.equation_classes.max_list import MaxList
import math

try:
    import compiler_gym  # noqa: F401
    from compiler_gym.spaces import Commandline, CommandlineFlag  # noqa
except ImportError:
    warnings.warn(message="CompilerGym not found. Proceeding without it.")


class GymGameState(GameState):
    def __init__(
        self, env, observation, production_action=None, previous_state=None, done=False
    ):
        super().__init__(
            syntax_tree=None,
            observation=observation,
            done=done,
            production_action=production_action,
            previous_state=previous_state,
        )
        self.env = env
        self.hash = str(observation["obs"])


class GymGame(Game):

    def __init__(self, args, env):
        super().__init__()
        self.args = args
        self.env = env
        self.max_list = MaxList(self.args)

        self.goals = (
            {}
            if self.args.maze_diverse_goals and self.env.spec.id.startswith("PointMaze")
            else None
        )

        self.env.reset(seed=args.seed)
        a_size = self.getActionSize()

        s = [f"S -> S [{1.0 / a_size}]\n"] * a_size  # uniform prior

        self.grammar = PCFG.fromstring("".join(s))
        self.max_path_length = self.env.spec.max_episode_steps
        add_prior(self.grammar, args)

    def getInitialState(self) -> GymGameState:
        if self.args.maze_diverse_goals and self.env.spec.id.startswith("PointMaze"):
            # force rarely selected goal cells
            goal = None
            # reset if goal cell has been selected in more than 10% cases
            while goal is None or self.goals.get(goal, 0) > 0.1 * sum(
                self.goals.values()
            ):
                obs, _ = self.env.reset()
                goal = str(
                    self.env.unwrapped.maze.cell_xy_to_rowcol(self.env.unwrapped.goal)
                )
            # update stats
            self.goals[goal] = self.goals.get(goal, 0) + 1
        else:
            obs, _ = self.env.reset()

        return GymGameState(None, {"last_symbol": "S", "obs": obs})

    def getDimensions(self) -> typing.Tuple[int, ...]:
        return self.env.observation_space.shape

    def getActionSize(self) -> int:
        return self.env.action_space.n.item()

    def getNextState(
        self, state: GymGameState, action: int, steps_done: int = 0, **kwargs
    ) -> typing.Tuple[GameState, float]:
        # avoid using deepcopy on env, reset to state (obs + goal) instead
        self.reset_env_to_state(state=state, steps_done=steps_done)

        obs, reward, terminated, truncated, __ = self.env.step(action)

        if reward != self.args.maximum_reward:
            # add random noise to reward if needed
            reward += (random.random() - 0.5) * self.args.reward_noise

        next_state = GymGameState(
            None,
            {"last_symbol": "S", "obs": obs},
            production_action=action,
            previous_state=state,
            done=terminated or truncated,
        )
        self.max_list.add(state=next_state, key=reward)
        next_state.reward = reward
        return next_state, reward

    def getLegalMoves(self, state: GameState) -> np.ndarray:
        if not state.done:
            return np.ones(self.getActionSize())
        else:
            return np.zeros(self.getActionSize())

    def getGameEnded(self, state: GameState, **kwargs) -> typing.Union[float, int]:
        return state.done

    def buildObservation(self, state: GameState) -> np.ndarray:
        return state.observation

    def getSymmetries(self, state: GameState, pi: np.ndarray) -> typing.List:
        pass

    def getHash(self, state: GameState) -> typing.Union[str, bytes, int]:
        return str(state)

    def reset_env_to_state(self, state, steps_done):
        self.env.reset()

        if self.env.spec.id.startswith("bitflip"):
            self.env._elapsed_steps = steps_done
            self.env.unwrapped.state = np.copy(state.observation["obs"]["state"])
            self.env.unwrapped.goal = np.copy(state.observation["obs"]["goal"])

        elif self.env.spec.id.startswith("PointMaze"):
            # we have 2 wrappers, 3rd one is TimeLimit
            self.env.env.env._elapsed_steps = steps_done  # TODO: rewrite this?
            self.env.unwrapped.data.qpos = np.copy(
                state.observation["obs"]["observation"][:2]
            )
            self.env.unwrapped.data.qvel = np.copy(
                state.observation["obs"]["observation"][2:]
            )
            self.env.unwrapped.goal = np.copy(state.observation["obs"]["desired_goal"])

        else:
            raise NotImplementedError()


def make_env(env_str: str, max_episode_steps):
    if env_str.startswith("PointMaze"):
        gym.register_envs(gymnasium_robotics)
        return NegativeRewardWrapper(
            DiscreteActionWrapper(
                gym.make(
                    env_str,
                    reward_type="sparse",
                    continuing_task=False,
                    reset_target=False,
                    max_episode_steps=max_episode_steps,
                )
            )
        )
    elif env_str == "CartPole-v1":
        return CartPoleWrapper(gym.make(env_str, max_episode_steps=max_episode_steps))
    elif env_str == "CliffWalking-v0":
        return CliffWrapper(gym.make(env_str, max_episode_steps=max_episode_steps))
    elif env_str.startswith("cbench"):
        CompilerGymWrapper.env = compiler_gym.make(
            "llvm-v0",  # compiler to use
            benchmark=env_str,  # program to compile
            observation_space="Autophase",  # observation space
            reward_space="IrInstructionCountOz",  # optimization target
            max_episode_steps=max_episode_steps,
        )
        return CompilerGymWrapper()
    elif env_str in list(gym.envs.registry.keys()):
        return gym.make(env_str)
    else:
        raise KeyError(f'Environment "{env_str}" does not exist!')


class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(DiscreteActionWrapper, self).__init__(env)
        self.action_space = gym.spaces.Discrete(8)

    def action(self, action):
        box_actions = [
            # without noop action [0, 0]
            [-1, 0],
            [0, -1],
            [0, 1],
            [1, 0],
            [-1 / np.sqrt(2), -1 / np.sqrt(2)],
            [-1 / np.sqrt(2), 1 / np.sqrt(2)],
            [1 / np.sqrt(2), -1 / np.sqrt(2)],
            [1 / np.sqrt(2), 1 / np.sqrt(2)],
        ]
        return np.array(box_actions[action])


class NegativeRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return 0 if reward == 1 else -1

    def compute_reward_(self, achieved_goal, desired_goal, info):
        r = self.compute_reward(achieved_goal, desired_goal, info)
        return 0 if r == 1 else -1


class CartPoleWrapper(gym.Wrapper):
    def step(self, action):
        obs, _, term, trunc, info = super().step(action)
        done = term or trunc
        return obs, -1.005 * done + 0.005, done, trunc, info


class CliffWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        done = term or trunc
        return obs, done + 1 + reward, done, trunc, info


class CompilerGymWrapper:
    max_list = MaxList(10)
    env = None

    def __init__(self):
        env = CompilerGymWrapper.env
        assert env.spec is not None
        self._max_episode_steps = env.spec.max_episode_steps
        self._elapsed_steps = None
        self.selected_actions = []
        terminal = CommandlineFlag(
            name="end-of-episode",
            flag="# end-of-episode",
            description="End the episode",
        )
        self.action_space = Commandline(
            items=[
                CommandlineFlag(
                    name=name,
                    flag=flag,
                    description=description,
                )
                for name, flag, description in zip(
                    env.action_space.names,
                    env.action_space.flags,
                    env.action_space.descriptions,
                )
            ]
            + [terminal],
            name=f"{type(self).__name__}<{env.action_space.name}>",
        )
        self.terminal_action: int = len(self.action_space.flags) - 1
        self.observation_space = env.observation_space
        self.reward_space = env.reward_space
        self.observation_space_spec = env.observation_space_spec
        self.spec = env.spec

    def reset(self, seed=None):  # noqa: F841
        self._elapsed_steps = 0
        self.selected_actions = []
        return self.selected_actions, {}

    def step(self, action):
        reward = [0.0]
        done = False
        trunc = False
        info = ""
        terminal_action_selected = action == self.terminal_action

        if not terminal_action_selected:
            self.selected_actions.append(action)

        if (
            len(self.selected_actions) >= self._max_episode_steps
        ) or terminal_action_selected:
            obs, reward, done, info = self.multistep(
                self.selected_actions,
                observation_spaces=[self.observation_space],
                reward_spaces=[self.reward_space],
                observations=[self.observation_space_spec],
                rewards=[self.reward_space],
            )
            if math.isfinite(reward[0]):
                CompilerGymWrapper.max_list.add(state=self.selected_actions, key=reward)

            if len(self.selected_actions) >= self._max_episode_steps:
                trunc = True
            else:
                trunc = False
        if terminal_action_selected:
            self.selected_actions.append(action)
            done = True

        return self.selected_actions, reward[-1], done, trunc, info

    def multistep(self, actions, **kwargs):
        env = CompilerGymWrapper.env
        env.reset()
        actions = list(actions)
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = env.multistep(actions, **kwargs)
        self._elapsed_steps += len(actions)
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True

        return observation, reward, done, info

    @staticmethod
    def close():
        CompilerGymWrapper.env.close()
