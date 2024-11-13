import typing
import tensorflow as tf
import numpy as np
from src.neural_nets.maze.maze_net import MazeNNet
from src.utils.logging import get_log_obj


def prepare_batch_obs(examples):
    return np.array(
        [
            np.concatenate(
                (
                    e["observation"]["obs"]["observation"],
                    e["observation"]["obs"]["desired_goal"],
                )
            )
            for e in examples
        ]
    )


def prepare_batch(examples):
    return (
        prepare_batch_obs(examples),
        np.array([e["probabilities_actor"] for e in examples]),
        np.array([e["observed_return"] for e in examples]),
    )


class MazeRulePredictorSkeleton:

    def __init__(self, game, args):
        self.game = game
        self.net = MazeNNet(game=game, args=args)
        self.args = args
        self.logger = get_log_obj(args=args, name="AlphaZeroRulePredictor")

    def train(self, examples: typing.List):
        """
        This function trains the neural network with data gathered from self-play.
        :param examples: a list of training examples of the form: (o_t, (pi_t, v_t), w_t)
        """
        obs, target_pis, target_vs = prepare_batch(examples)

        _, pi_loss, v_loss = self.net.model.train_on_batch(
            x=obs, y=[target_pis, target_vs]
        )
        return (
            pi_loss,
            v_loss,
            0,
        )

    def predict(self, examples):
        obs = prepare_batch_obs(examples)
        pi, v = self.tf_predict(obs)
        return pi.numpy()[0], v.numpy()[0][0]

    @tf.function
    def tf_predict(self, obs):
        pi, v = self.net.model(obs, training=False)
        return pi, v
