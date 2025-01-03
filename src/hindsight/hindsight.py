import random
import numpy as np
from src.game.find_equation_game import FindEquationGame
from src.game.game_history import GameHistory
import copy
from src.equation_modules.generate_datasets.dataset_generator import (
    constant_dict_to_string,
)
from src.game.gym_game import GymGame
from src.utils.logging import get_log_obj


class Hindsight:
    """
    Hindsight Experience Replay for the last played episode.

    :param seed: RNG seed
    :param game: Game instance
    :param episode_history: Last episode history (played by the agent)
    :param mcts: MCTS search tree constructed during last episode
    :param gamma: Discount factor for hindsight rewards

    :param trajectory_selection: Trajectory choice: played (=episode_history) or random from the MCTS tree
    :param num_trajectories: Number of trajectories to use (only for random trajectory_selection)
    :param num_samples: Maximum number of goals to sample for each observation
    :param policy: Policy target choice: original from MCTS probabilities, one-hot or one-hot with random noise

    :param goal_selection: Strategy for selecting goals: future visited episode states or final state
    :param aggressive_returns_lambda: Multiplication factor for hindsight rewards (returns in our case)
        to attempt countering created bias: http://arxiv.org/abs/1809.02070
    :param experience_ranking: Filtering of virtual goals far away from the original:
        https://ieeexplore.ieee.org/abstract/document/8850705/
    :param experience_ranking_threshold: Maximum distance between virtual and real goals allowed

    :param reward_noise: Amount of noise added to rewards to test learning stability
    :param logging_level: Importance of messages required for logging
    """

    def __init__(
        self,
        seed,
        game,
        episode_history,
        mcts,
        gamma,
        trajectory_selection,
        num_trajectories,
        num_samples,
        policy,
        goal_selection,
        aggressive_returns_lambda=1,
        experience_ranking=False,
        experience_ranking_threshold=1,
        reward_noise=0,
        logging_level=30,
    ):
        self.game = game
        self.episode_history = episode_history
        self.mcts = mcts
        self.gamma = gamma

        self.trajectory_selection = trajectory_selection
        self.num_trajectories = num_trajectories
        self.num_samples = num_samples
        self.policy = policy
        self.goal_selection = goal_selection

        self.aggressive_returns_lambda = aggressive_returns_lambda
        self.experience_ranking = experience_ranking
        self.experience_ranking_threshold = experience_ranking_threshold

        self.reward_noise = reward_noise

        self.logger = get_log_obj(args=None, name="Hindsight", level=logging_level)

        self.random = random.Random(x=seed)
        self.np_random = np.random.default_rng(seed=seed)

    def create_hindsight_samples(self):
        """
        Creates HER samples for the last played episode.

        :return: A list of GameHistory objects containing HER samples.
        """
        if self.trajectory_selection == "played":
            # use actually played episode history
            return self.create_trajectory_hindsight_samples(
                trajectory=self.episode_history
            )
        elif self.trajectory_selection == "mcts_random":
            hindsight_histories = []
            # get terminal states from mcts tree
            terminal_states = self.get_mcts_terminal_states()
            if len(terminal_states) < self.num_trajectories:
                return []
            # if possible, construct path to each of them from root
            trajectories = [
                self.construct_trajectory_to_state(final_state=s)
                for s in self.random.sample(terminal_states, self.num_trajectories)
            ]
            # add hindsight using constructed trajectories
            for t in trajectories:
                hindsight_histories.extend(
                    self.create_trajectory_hindsight_samples(trajectory=t)
                )
        else:
            raise NotImplementedError()

        return hindsight_histories

    def create_trajectory_hindsight_samples(self, trajectory):
        """
        Creates HER samples using specified trajectory of the last played episode.

        For each trajectory state, min(#possible goals, num_samples) additional goals are sampled and resulting
        hindsight states created.

        GameHistories are used as a convenient container for samples and do NOT represent state sequences.
        Each i-th element in n-th history is a hindsight sample for n-th sampled goal for i-th observation
        in the trajectory. Empty histories (caused by #possible goals < num_states in some cases) are filtered out.

        :param trajectory: Trajectory (GameHistory) used
        :return: A list of GameHistory objects containing HER samples.
        """

        # create hindsight history for each sampled goal
        hindsight_histories = [GameHistory() for _ in range(self.num_samples)]
        # for each observation in episode
        for i in range(len(trajectory.observations) - 1):  # without the final one
            goal_indices, goal_observations = self.sample_goals(
                episode_observations=trajectory.observations,
                i=i,
            )
            # for each virtual goal
            for g in range(len(goal_observations)):
                # change original observation goal
                relabeled_observation = self.relabel_observation_goal(
                    observation=trajectory.observations[i],
                    hindsight_goal_observation=goal_observations[g],
                    syntax_tree=(
                        trajectory.syntax_tree
                        if isinstance(self.game, FindEquationGame)
                        else None
                    ),
                )
                if relabeled_observation is None:
                    continue  # infinite y or invalid value for equations
                # choose policy to use
                if self.policy == "original":
                    hindsight_policy = trajectory.probabilities[i]
                elif self.policy == "one_hot" or self.policy == "one_hot_noisy":
                    hindsight_policy = self.get_one_hot_policy(
                        episode_actions=trajectory.actions, index=i
                    )
                else:
                    raise NotImplementedError()
                # save relabeled observation and policy to hindsight history
                hindsight_histories[g].capture_with_observation(
                    observation=relabeled_observation,
                    pi=hindsight_policy,
                    action=None,
                    r=0,
                    v=0,
                )
                # calculate and save new return
                hindsight_histories[g].observed_returns.append(
                    self.compute_hindsight_return(
                        observation_index=i,
                        episode_observations=trajectory.observations,
                        goal_index=goal_indices[g],
                        goal_observation=goal_observations[g],
                    )
                )
        # filter out empty histories
        hindsight_histories = [
            h for h in hindsight_histories if len(h.observations) > 0
        ]
        # define searched and found equations
        if isinstance(self.game, FindEquationGame):
            for h in hindsight_histories:
                h.found_equation = h.searched_equation = h.observations[0][
                    "true_equation"
                ]
        return hindsight_histories

    def sample_goals(self, episode_observations, i):
        """
        Samples goal states according to the selection strategy.

        :param episode_observations: List of observations
        :param i: Current observation index
        :return: A list of tuples (i,o) with chosen goal indices and observations.
        """
        # find all possible goals depending on chosen strategy
        indexed_observations = [
            (index, observation)
            for index, observation in enumerate(episode_observations)
        ]
        if self.goal_selection == "future":
            possible_goals = indexed_observations[i + 1 :]  # noqa
        elif self.goal_selection == "final":
            possible_goals = [indexed_observations[-1]]
        else:
            raise NotImplementedError()

        # skip goals too far away from original if needed
        if self.experience_ranking:
            possible_goals = [
                g
                for g in possible_goals
                if self.compute_distance_to_original_goal(virtual_goal_observation=g[1])
                <= self.experience_ranking_threshold
            ]

        # sample
        if len(possible_goals) > 0:
            return zip(
                *self.random.sample(
                    population=possible_goals,
                    # sample fewer goals if we do not have enough unique observations
                    k=min(len(possible_goals), self.num_samples),
                )
            )
        else:
            return [], []

    def compute_hindsight_return(
        self, observation_index, episode_observations, goal_index, goal_observation
    ):
        """
        Computes total hindsight return from current state.

        :param observation_index: Current observation index
        :param episode_observations: List of observations
        :param goal_index: Goal observation index
        :param goal_observation: Goal observation

        :return: Total hindsight return from current state.
        """
        hindsight_rewards = []
        # for each future episode observation until virtual goal
        for j in range(observation_index, goal_index):
            # compute and save new reward
            if isinstance(self.game, FindEquationGame):
                hindsight_rewards.append(
                    self.game.args.maximum_reward if j + 1 == goal_index else 0
                )
            else:
                hindsight_rewards.append(
                    self.get_reward_with_goal(
                        # current reward is calculated using next observation -> j + 1
                        observation=episode_observations[j + 1],
                        goal_observation=goal_observation,
                    )
                )
                # stop if goal is already reached
                if hindsight_rewards[-1] == self.game.args.maximum_reward:
                    break
                else:
                    # add random noise to reward if needed
                    hindsight_rewards[-1] += (
                        self.random.random() - 0.5
                    ) * self.reward_noise
        # calculate total return for state (and multiply by lambda if aggressive rewards are used)
        return (
            sum(
                [
                    np.power(self.gamma, k) * hindsight_rewards[k]
                    for k in range(len(hindsight_rewards))
                ]
            )
            * self.aggressive_returns_lambda
        )

    def relabel_observation_goal(
        self, observation, hindsight_goal_observation, syntax_tree=None
    ):
        """
        Constructs a copy of current observation with relabeled goal state.

        :param observation: Current observation
        :param hindsight_goal_observation: Goal observation

        In case of FindEquationGame we also need:
        :param syntax_tree: Syntax tree of the last episode

        :return: A copy of current observation with relabeled goal state.
        """
        relabeled_observation = copy.deepcopy(observation)
        if isinstance(self.game, FindEquationGame):
            try:
                y_calc = syntax_tree.evaluate_subtree(
                    node_id=syntax_tree.start_node.node_id,
                    dataset=relabeled_observation["data_frame"],
                )
                if np.all(np.isfinite(y_calc)):
                    c_string_backward = constant_dict_to_string(syntax_tree)
                    equation_string = (
                        f"{syntax_tree.rearrange_equation_infix_notation(-1)[1]}"
                    )
                    relabeled_observation["data_frame"]["y"] = y_calc
                    relabeled_observation["true_equation"] = (
                        f"{equation_string}_{c_string_backward}"
                    )
                    relabeled_observation["true_equation_hash"] = (
                        equation_string.strip()
                    )
                    return relabeled_observation
                else:
                    self.logger.warning("infinite value in y calculation")
                    return None
            except Exception as e:
                self.logger.warning(
                    "y calculation failed: " + getattr(e, "message", repr(e))
                )
                return None  # invalid value in y calculation
        elif isinstance(self.game, GymGame):
            relabeled_observation["obs"]["desired_goal"] = hindsight_goal_observation[
                "obs"
            ]["achieved_goal"]
            return relabeled_observation
        else:
            raise NotImplementedError()

    def get_reward_with_goal(self, observation, goal_observation=None):
        """
        Gets a reward signal for current observation (with hindsight goal) from the environment.

        :param observation: Current observation
        :param goal_observation: Goal observation
        :return: Game reward for current observation with specified goal.
        """
        if isinstance(self.game, GymGame):
            reward = self.game.env.unwrapped.compute_reward(
                observation["obs"]["achieved_goal"],
                goal_observation["obs"]["achieved_goal"],
                {},
            )
            if self.game.env.spec.id.startswith("PointMaze"):
                # NegativeRewardWrapper is not applied here,
                # so we need -1 for binary negative rewards
                return reward - 1
            else:
                return reward
        else:
            raise NotImplementedError()

    def compute_distance_to_original_goal(self, virtual_goal_observation):
        """
        Calculates a measure of distance between virtual and real goals. Implementation details are
        environment-specific.

        :param virtual_goal_observation: Virtual goal observation
        :return: Distance between virtual and real goals.
        """
        if isinstance(self.game, GymGame):
            return np.linalg.norm(
                virtual_goal_observation["obs"]["achieved_goal"]
                - virtual_goal_observation["obs"]["desired_goal"]
            )
        else:
            raise NotImplementedError()

    def get_one_hot_policy(self, episode_actions, index):
        """
        Constructs one-hot move policy for current state
        (probability of actually selected action is set to 1).

        :param episode_actions: List of episode actions
        :param index: Current state index
        :return: One-hot move policy for current state.
        """
        one_hot = np.zeros(self.game.getActionSize())
        one_hot[episode_actions[index]] = 1
        if self.policy == "one_hot_noisy":
            one_hot += self.np_random.random(one_hot.shape) * 1e-2
            # normalize to create a valid distribution
            one_hot = np.divide(one_hot, np.sum(one_hot))
        return one_hot

    def construct_trajectory_to_state(self, final_state):
        """
        Constructs a trajectory from MCTS tree root to a given terminal state
        as a GameHistory.

        :param final_state: Trajectory terminal state
        :return: A GameHistory containing trajectory from MCTS tree root to a given terminal state.
        """
        episode_history = GameHistory()
        current_state = final_state
        # final observation and syntax tree
        episode_history.observations.append(final_state.observation)
        episode_history.syntax_tree = (
            final_state.syntax_tree if isinstance(self.game, FindEquationGame) else None
        )
        # repeat until the first state
        while current_state.previous_state:
            # get previous state
            previous_state = current_state.previous_state
            # find action visit counts from it
            previous_state_actions = np.nonzero(
                self.mcts.valid_moves_for_s[previous_state.hash]
            )[0]
            previous_state_action_visits = np.zeros(self.game.getActionSize())
            for action in previous_state_actions:
                key = (previous_state.hash, action)
                if key in self.mcts.times_edge_s_a_was_visited:
                    previous_state_action_visits[action] = (
                        self.mcts.times_edge_s_a_was_visited[key]
                    )
            # use them to construct a probability distribution
            probabilities = previous_state_action_visits / np.sum(
                previous_state_action_visits
            )
            # set previous state chosen action
            previous_state.action = current_state.production_action
            # capture previous state to history
            episode_history.capture(
                state=previous_state, pi=probabilities, r=current_state.reward, v=0
            )
            # move backwards
            current_state = previous_state

        # reverse order of states in history (we started from final) and compute return
        episode_history.reverse()
        episode_history.compute_returns(self.gamma)
        # close history
        episode_history.terminated = True
        return episode_history

    def get_mcts_terminal_states(self):
        """
        Returns a list of all terminal states in a search tree. For FindEquationGame
        we have to make sure syntax trees are complete.

        :return: A list of all terminal states in a search tree.
        """
        if isinstance(self.game, FindEquationGame):
            return [
                self.mcts.Ssa[key]
                for key in list(self.mcts.Ssa.keys())
                if self.mcts.Ssa[key].syntax_tree.complete
            ]
        else:
            return [
                self.mcts.Ssa[key]
                for key in list(self.mcts.Ssa.keys())
                if self.mcts.Ssa[key].done
            ]
