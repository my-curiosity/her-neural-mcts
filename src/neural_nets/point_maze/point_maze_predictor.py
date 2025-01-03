import tensorflow as tf
import numpy as np
from src.neural_nets.point_maze.point_maze_net import PointMazeNet
from src.utils.logging import get_log_obj


def prepare_batch_obs(examples):
    """
    Concatenates state observation and goal in example data.
    :param examples:
    """
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
    """
    Transforms self-play example data to NN input shape.
    :param examples:
    """
    return (
        prepare_batch_obs(examples),
        np.array([e["probabilities_actor"] for e in examples]),
        np.array([e["observed_return"] for e in examples]),
    )


class PointMazePredictor:

    def __init__(self, game, args):
        self.game = game
        self.net = PointMazeNet(game=game, args=args)
        self.args = args
        self.logger = get_log_obj(args=args, name="AlphaZeroRulePredictor")

    def train(self, examples):
        """
        Trains the neural network on examples.
        :param examples:
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
        """
        Predicts move probabilities and state value using examples.
        :param examples:
        """
        obs = prepare_batch_obs(examples)
        pi, v = self.tf_predict(obs)
        return pi.numpy()[0], v.numpy()[0][0]

    @tf.function
    def tf_predict(self, obs):
        """
        Helper function to speedup tensorflow model prediction
        :param obs: observations to use
        """
        pi, v = self.net.model(obs, training=False)
        return pi, v
