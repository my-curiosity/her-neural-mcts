import random
import typing
import gymnasium as gym
import gymnasium_robotics
import numpy as np
from pcfg import PCFG
from src.utils.get_grammar import add_prior
from src.game.game import Game, GameState
from src.equation_classes.max_list import MaxList


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

        self.goals = {}
        if self.env.spec.id.startswith("PointMaze"):
            cells = [
                self.env.unwrapped.maze.cell_xy_to_rowcol(c).tolist()
                for c in self.env.unwrapped.maze.unique_goal_locations
            ]
            # 1/4 cells are blocked
            self.blocked_goals = random.sample(cells, int(len(cells) / 4))
        else:  # bitflip
            self.blocked_goals = [
                [random.getrandbits(1) for _ in range(self.args.bitflip_num_bits)]
                for _ in range(10000)
            ]

        self.env.reset(seed=args.seed)
        a_size = self.getActionSize()

        s = [f"S -> S [{1.0 / a_size}]\n"] * a_size  # uniform prior

        self.grammar = PCFG.fromstring("".join(s))
        self.max_path_length = self.env.spec.max_episode_steps
        add_prior(self.grammar, args)

    def getInitialState(self, mode="train") -> GymGameState:
        if mode == "test":
            if self.args.test_generalization == "unseen":
                # force unseen goals only
                goal = random.choice(self.blocked_goals)
            elif self.args.test_generalization == "rare":
                # force least frequent goals
                goal = np.array(sorted(self.goals.items(), key=lambda g: g[1])[0][0])
                self.goals[tuple(goal)] += 1  # update stats
            else:
                raise NotImplementedError
            # reset env with chosen goal
            if self.env.spec.id.startswith("PointMaze"):
                obs, _ = self.env.reset(options={"goal_cell": goal})
            else:  # bitflip
                obs, _ = self.env.reset(options={"goal": goal})

        else:  # train
            goal = None
            while goal is None or (
                self.args.test_generalization == "unseen" and goal in self.blocked_goals
            ):
                obs, _ = self.env.reset()
                goal = (
                    self.env.unwrapped.maze.cell_xy_to_rowcol(self.env.unwrapped.goal)
                    if self.env.spec.id.startswith("PointMaze")
                    else obs["goal"]
                ).tolist()
            # update stats
            self.goals[tuple(goal)] = self.goals.get(tuple(goal), 0) + 1

        return GymGameState(None, {"last_symbol": "S", "obs": obs})

    def getDimensions(self) -> typing.Tuple[int, ...]:
        return self.env.observation_space.shape

    def getActionSize(self) -> int:
        return self.env.action_space.n.item()

    def getNextState(
        self, state: GymGameState, action: int, steps_done: int = 0, **kwargs
    ) -> typing.Tuple[GameState, float]:
        # avoid using deepcopy on env, reset to state (obs + goal) instead
        reset_env_to_state(env=self.env, state=state, steps_done=steps_done)

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


def reset_env_to_state(env, state, steps_done):
    env.reset()

    if env.spec.id.startswith("bitflip"):
        env._elapsed_steps = steps_done
        env.unwrapped.state = np.copy(state.observation["obs"]["state"])
        env.unwrapped.goal = np.copy(state.observation["obs"]["goal"])

    elif env.spec.id.startswith("PointMaze"):
        # we have 2 wrappers, 3rd one is TimeLimit
        env.env.env._elapsed_steps = steps_done  # TODO: rewrite this?
        env.unwrapped.data.qpos = np.copy(state.observation["obs"]["observation"][:2])
        env.unwrapped.data.qvel = np.copy(state.observation["obs"]["observation"][2:])
        env.unwrapped.goal = np.copy(state.observation["obs"]["desired_goal"])
        env.unwrapped.update_target_site_pos()

    else:
        raise NotImplementedError()
