"""
Define the base self-play/ data gathering class. This class should work with any MCTS-based neural network learning
algorithm like AlphaZero or MuZero. Self-play, model-fitting, and pitting is performed sequentially on a single-thread
in this default implementation.

Notes:
 - Code adapted from https://github.com/suragnair/alpha-zero-general
 - Base implementation done.
 - Base implementation sufficiently abstracted to accommodate both AlphaZero and MuZero.
 - Documentation 15/11/2020
"""

import os
import typing
from pickle import Pickler, Unpickler, HIGHEST_PROTOCOL
from collections import deque
from abc import ABC
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from moviepy import VideoFileClip, concatenate_videoclips

import numpy as np
from tqdm import trange

from src.game.find_equation_game import FindEquationGame
from src.game.game_history import GameHistory, sample_batch
from datetime import datetime
import tensorflow as tf
import wandb

from src.game.gym_game import (
    DiscreteActionWrapper,
    NegativeRewardWrapper,
    reset_env_to_state,
    GymGameState,
)
from src.utils.logging import get_log_obj
from src.utils.files import highest_number_in_files
from definitions import ROOT_DIR
from src.hindsight.hindsight import Hindsight


class Coach(ABC):
    """
    This class controls the self-play and learning loop.
    """

    def __init__(
        self,
        game,
        rule_predictor,
        args,
        search_engine,
        run_name,
        checkpoint_train,
        checkpoint_manager,
        checkpoint_test=None,
    ) -> None:
        """
        Initialize the self-play class with an environment, an agent to train, requisite hyperparameters, a MCTS search
        engine, and an agent-interface.
        # :param rule_predictor_test:
        # :param game_test:
        :param run_name:
        :param game: Game Implementation of Game class for environment logic.
        :param rule_predictor: Some implementation of a neural network class to be trained.
        :param args: Data structure containing parameters for self-play.
        :param search_engine: Class containing the logic for performing MCTS using the neural_net.
        """

        self.metrics_test = None
        self.metrics_train = None
        self.game = game
        self.args = args

        # Initialize replay buffer and helper variable
        self.trainExamplesHistory = deque(
            maxlen=self.args.selfplay_buffer_window
            * self.args.num_selfplay_iterations
            * (1 + self.args.hindsight_samples)
        )

        # Initialize network and search engine
        self.rule_predictor = rule_predictor
        self.mcts = search_engine(self.game, self.args, self.rule_predictor)

        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

        self.log_dir = f"{ROOT_DIR}/out/logs/{run_name}"
        self.file_writer = tf.summary.create_file_writer(self.log_dir + "/metrics")
        self.file_writer.set_as_default()
        self.checkpoint = checkpoint_train
        self.checkpoint_manager = checkpoint_manager
        self.checkpoint_test = checkpoint_test
        self.logger = get_log_obj(args=args, name="coach")
        self.logger_test = get_log_obj(args=args, name="coach_test")

    @staticmethod
    def getCheckpointFile(iteration: int) -> str:
        """Helper function to format model checkpoint filenames"""
        return f"checkpoint_{iteration}.pth.tar"

    def sampleBatch(
        self, histories: typing.List[GameHistory], batch_i: int
    ) -> typing.List:
        """
          Sample a batch of data from the current replay buffer (with or without prioritization).
        Construct a batch of data-targets for gradient optimization of the AlphaZero neural network.

        The procedure samples a list of game and inside-game coordinates of length 'batch_size'. This is done either
        uniformly or with prioritized sampling. Using this list of coordinates, we sample the according games, and
        the according points of times within the game to generate neural network inputs, targets, and sample weights.

        The targets for the neural network consist of MCTS move probability vectors and TD/ Monte-Carlo returns.

        Optionally uses (Hindsight-) Combined Experience Replay (https://www.researchgate.net/publication/346030781)
        to guarantee that latest episode transitions are included in the batch

        :param histories: List of GameHistory objects. Contains all game-trajectories in the replay-buffer.
        :param batch_i: Index of the batch being selected in current training iteration. Required for CHER.
        :return: List of training examples: (observations, (move-probabilities, TD/ MC-returns), sample_weights)
        """

        # remove final observation in non-hindsight histories if necessary
        for h in histories:
            if len(h.observations) > len(h.probabilities):
                h.observations = h.observations[:-1]

        # Generate coordinates within the replay buffer to sample from. Also generate the loss scale of said samples.
        sample_coordinates, sample_weight = sample_batch(
            list_of_histories=histories,
            n=self.args.batch_size_training,
            prioritize=self.args.prioritize,
            alpha=self.args.prioritize_alpha,
            beta=self.args.prioritize_beta,
        )

        if self.args.hindsight_combined_experience_replay:
            if len(histories[-1]) >= batch_i + 1:
                # C(H)ER: add i-th transition starting from episode end to the batch
                # expects real episode history to be saved AFTER hindsight samples
                sample_coordinates.append((-1, -(batch_i + 1)))
                # just to keep both lists equally long
                sample_weight = np.append(sample_weight, 1)

        # Collect training examples for AlphaZero: (o_t, (pi_t, v_t), w_t)
        examples = [
            {
                "observation": histories[h_i].stackObservations(length=1, t=i),
                "probabilities_actor": histories[h_i].probabilities[i],
                "observed_return": histories[h_i].observed_returns[i],
                "loss_scale": loss_scale,
                "found_equation": (
                    histories[h_i].found_equation
                    if isinstance(self.game, FindEquationGame)
                    else None
                ),
            }
            for (h_i, i), loss_scale in zip(sample_coordinates, sample_weight)
        ]
        return examples

    def execute_one_game(self, game, mcts, mode) -> GameHistory:
        """
        Perform one episode of self-play for gathering data to train neural networks on.

        The implementation details of the neural networks/ agents, temperature schedule, data storage
        is kept highly transparent on this side of the algorithm. Hence, for implementation details
        see the specific implementations of the function calls.

        At every step we record a snapshot of the state into a GameHistory object, this includes the observation,
        MCTS search statistics, performed action, and observed rewards. After the end of the episode, we close the
        GameHistory object and compute internal target values.

        :return: GameHistory Data structure containing all observed states and statistics required for network training.
        """
        # Update MCTS visit count temperature according to an episode or weight update schedule.
        temp = self.get_temperature(game)

        history = GameHistory()
        # Always from perspective of player 1 for boardgames.
        state = game.getInitialState(mode)

        formula_started_from = (
            state.observation["current_tree_representation_str"]
            if isinstance(game, FindEquationGame)
            else None
        )

        i = 0
        while not state.done:
            # Compute the move probability vector and state value using MCTS for the current state of the environment.
            pi, v = mcts.run_mcts(
                state=state,
                num_mcts_sims=self.args.num_MCTS_sims,
                temperature=temp,
                depth=i,
            )
            # Take a step in the environment and observe the transition and store necessary statistics.
            # TODO: greedy choice only in test?
            state.action = np.argmax(pi)  # np.random.choice(len(pi), p=pi)

            next_state, r = game.getNextState(
                state=state, action=state.action, steps_done=i
            )

            history.capture(state=state, pi=pi, r=r, v=v)
            # Update state of control
            state = next_state
            i += 1

        history.observations.append(state.observation)  # final observation
        history.syntax_tree = (
            state.syntax_tree if isinstance(game, FindEquationGame) else None
        )

        game.close(state)
        history.terminate(
            formula_started_from=(
                formula_started_from if isinstance(game, FindEquationGame) else None
            ),
            found_equation=(
                state.syntax_tree.rearrange_equation_infix_notation(-1)[1]
                if isinstance(game, FindEquationGame)
                else None
            ),
        )
        history.compute_returns(gamma=self.args.gamma)
        return history

    def get_temperature(self, game):
        try:
            temp = self.args.temp_0 * np.exp(
                self.args.temperature_decay * np.float32(self.checkpoint.step.numpy())
            )
        except FloatingPointError:
            temp = self.args.temp_0
        return temp

    def learn(self) -> None:
        """
        Control the data gathering and weight optimization loop. Perform 'num_selfplay_iterations' iterations
        of self-play to gather data, each of 'num_episodes' episodes. After every self-play iteration, train the
        neural network with the accumulated data. If specified, the previous neural network weights are evaluated
        against the newly fitted neural network weights, the newly fitted weights are then accepted based on some
        specified win/ lose ratio. Neural network weights and the replay buffer are stored after every iteration.
        Note that for highly granular vision based environments, that the replay buffer may grow to large sizes.
        """
        self.logger.warning(
            f"Starting with hindsight ({self.args.hindsight_samples} samples) ..."
        )

        self.metrics_train = {
            "mode": "train",
            "reward": tf.keras.metrics.Mean(dtype=tf.float32),
            "return": tf.keras.metrics.Mean(dtype=tf.float32),
            "solved": tf.keras.metrics.Mean(dtype=tf.float32),
        }
        self.metrics_test = {
            "mode": "test",
            "reward": tf.keras.metrics.Mean(dtype=tf.float32),
            "return": tf.keras.metrics.Mean(dtype=tf.float32),
            "solved": tf.keras.metrics.Mean(dtype=tf.float32),
        }
        if self.args.load_pretrained:
            self.loadTrainExamples(int(self.checkpoint.step))
        while self.checkpoint.step < self.args.max_iteration_to_run:
            self.logger.warning(
                f"------------------ITER"
                f" {int(self.checkpoint.step)}----------------"
            )
            # Self-play/ Gather training data.
            self.gather_data(
                metrics=self.metrics_train,
                mcts=self.mcts,
                game=self.game,
                num_selfplay_iterations=self.args.num_selfplay_iterations,
            )
            self.saveTrainExamples(int(self.checkpoint.step))

            save_path = self.checkpoint_manager.save(check_interval=True)
            self.logger.debug(
                f"Saved checkpoint for epoch {int(self.checkpoint.step)}: {save_path}"
            )

            test_now = (
                self.args.test_generalization != "off"
                and self.checkpoint.step % self.args.test_every_n_steps == 1
            )

            if test_now:
                self.gather_data(
                    metrics=self.metrics_test,
                    mcts=self.mcts,
                    game=self.game,
                    num_selfplay_iterations=self.args.num_selfplay_iterations_test,
                )

            for m in [self.metrics_train, self.metrics_test]:
                if m["mode"] == "train" or test_now:
                    wandb.log(
                        {
                            f"iteration": self.checkpoint.step,
                            f"reward_{m['mode']}": m["reward"].result(),
                            f"return_{m['mode']}": m["return"].result(),
                            f"solved_{m['mode']}": m["solved"].result(),
                        }
                    )

            self.checkpoint.step.assign_add(1)

    def update_network(self):
        # Backpropagation
        pi_loss, v_loss = 0, 0
        for i in range(self.args.num_gradient_steps):
            batch = self.sampleBatch(
                histories=list(self.trainExamplesHistory), batch_i=i
            )
            pi_batch_loss, v_batch_loss, _ = self.rule_predictor.train(batch)
            pi_loss += pi_batch_loss
            v_loss += v_batch_loss
        wandb.log(
            {
                f"iteration": self.checkpoint.step,
                f"Pi loss": pi_loss / self.args.num_gradient_steps,
                "V loss": v_loss / self.args.num_gradient_steps,
            }
        )

    def gather_data(self, metrics, mcts, game, num_selfplay_iterations):
        metrics["reward"].reset_state()
        metrics["return"].reset_state()
        metrics["solved"].reset_state()
        video_dir = os.getcwd() + "/video/"
        video_paths = []

        for i in trange(
            num_selfplay_iterations,
            desc=(
                "Playing episodes" if metrics["mode"] == "train" else "Testing episodes"
            ),
        ):
            mcts.clear_tree()

            episode_history = self.execute_one_game(
                game=game, mcts=mcts, mode=metrics["mode"]
            )

            metrics["reward"].update_state(
                game.max_list.max_list_state[-1].reward
                if len(game.max_list.max_list_state) > 0
                else -1
            )
            metrics["return"].update_state(episode_history.observed_returns[0])
            metrics["solved"].update_state(
                100 if self.episode_solved(episode_history) else 0
            )

            # record episode video for better visualization
            if self.args.record_video and self.args.game == "maze":
                video_paths.append(
                    self.record_game_video(
                        video_dir, episode_history, i, metrics["mode"]
                    )
                )

            if metrics["mode"] == "train":
                # add hindsight histories to ER
                if self.args.hindsight_samples > 0:
                    hindsight = Hindsight(
                        seed=self.args.seed,
                        game=game,
                        mcts=mcts,
                        gamma=self.args.gamma,
                        episode_history=episode_history,
                        # configuration
                        num_samples=self.args.hindsight_samples,
                        policy=self.args.hindsight_policy,
                        goal_selection=self.args.hindsight_goal_selection,
                        trajectory_selection=self.args.hindsight_trajectory_selection,
                        num_trajectories=self.args.hindsight_num_trajectories,
                        # advanced options
                        aggressive_returns_lambda=self.args.hindsight_aggressive_returns_lambda,
                        experience_ranking=self.args.hindsight_experience_ranking,
                        experience_ranking_threshold=self.args.hindsight_experience_ranking_threshold,
                        args=self.args,
                    )
                    self.trainExamplesHistory.extend(
                        hindsight.create_hindsight_samples()
                    )

                # add real history to ER (at the end to access last state transition easily)
                self.trainExamplesHistory.append(episode_history)

                if self.checkpoint.step > self.args.cold_start_iterations:
                    self.update_network()

        # save and upload a concatenation of all game videos from this iteration
        if self.args.record_video and self.args.game == "maze":
            self.save_iteration_video(video_dir, video_paths, metrics["mode"])

    def episode_solved(self, episode_history):
        if isinstance(self.game, FindEquationGame):
            return abs(episode_history.rewards[-1] - self.args.maximum_reward) < 0.02
        else:  # gym
            return episode_history.rewards[-1] == self.args.maximum_reward

    def saveTrainExamples(self, iteration: int) -> None:
        """
        Store the current accumulated data to a compressed file using pickle. Note that for highly dimensional
        environments, that the stored files may be considerably large and that storing/ loading the data may
        introduce a significant bottleneck to the runtime of the algorithm.
        :param iteration: int Current iteration of the self-play. Used as indexing value for the data filename.
        """
        folder = (
            ROOT_DIR
            / "saved_models"
            / self.args.data_path
            / str(self.args.experiment_name)
            / str(self.args.seed)
        )

        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = folder / f"buffer_{iteration}.examples"
        with open(filename, "wb+") as f:
            Pickler(f, protocol=HIGHEST_PROTOCOL).dump(self.trainExamplesHistory)

        # Don't hog up storage space and clean up old (never to be used again) data.
        old_checkpoint = folder / f"buffer_{iteration - 1}.examples"
        if os.path.isfile(old_checkpoint):
            os.remove(old_checkpoint)

    def loadTrainExamples(self, iteration: int) -> None:
        """
        Load in a previously generated replay buffer from the path specified in the .json arguments.
        """
        if len(self.args.replay_buffer_path) >= 1:
            if os.path.isfile(self.args.replay_buffer_path):
                with open(self.args.replay_buffer_path, "rb") as f:
                    self.logger.info(
                        f"Replay buffer {self.args.replay_buffer_path}  found. Read it."
                    )
                    self.trainExamplesHistory = Unpickler(f).load()
            else:
                self.logger.info(f"No replay buffer found. Use empty one.")
        else:
            folder = (
                ROOT_DIR
                / "saved_models"
                / self.args.data_path
                / str(self.args.experiment_name)
                / str(self.args.seed)
            )
            buffer_number = highest_number_in_files(path=folder, stem="buffer_")
            filename = folder / f"buffer_{buffer_number}.examples"

            if os.path.isfile(filename):
                with open(filename, "rb") as f:
                    self.logger.info(f"Replay buffer {buffer_number}  found. Read it.")
                    self.trainExamplesHistory = Unpickler(f).load()
            else:
                self.logger.info(f"No replay buffer found. Use empty one.")

    def record_game_video(self, video_dir, episode_history, idx, mode):
        video_prefix = (
            ("solved" if self.episode_solved(episode_history) else "unsolved")
            + "_iter"
            + str(int(self.checkpoint.step))
            + ("_test" if mode == "test" else "_train")
            + "_game"
            + str(idx)
        )
        video_env = NegativeRewardWrapper(
            DiscreteActionWrapper(
                RecordVideo(
                    env=gym.make(
                        "PointMaze_Medium-v3",
                        reward_type="sparse",
                        continuing_task=False,
                        reset_target=False,
                        max_episode_steps=self.args.max_episode_steps,
                        render_mode="rgb_array",
                    ),
                    video_folder=video_dir,
                    name_prefix=video_prefix,
                    episode_trigger=lambda x: True,
                )
            )
        )
        # reset env to starting state and execute chosen actions
        reset_env_to_state(
            video_env,
            GymGameState(None, episode_history.observations[0]),
            0,
        )
        for j in range(len(episode_history.actions)):
            action = episode_history.actions[j]
            obs, reward, terminated, truncated, _ = video_env.step(action)
            video_env.render()
            if terminated or truncated:
                break
        video_env.close()
        # return video path
        return video_dir + video_prefix + "-episode-0.mp4"

    def save_iteration_video(self, video_dir, video_paths, mode):
        iter_video = concatenate_videoclips([VideoFileClip(p) for p in video_paths])
        iter_video_path = (
            video_dir
            + "iter"
            + str(int(self.checkpoint.step))
            + ("_test" if mode == "test" else "_train")
            + ".mp4"
        )
        iter_video.write_videofile(iter_video_path)
        wandb.log({"video": wandb.Video(iter_video_path, format="mp4")})
