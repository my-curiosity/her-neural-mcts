import typing
import tensorflow as tf
from src.neural_nets.bitflip.bit_flip_net import BitFlipNNet
import numpy as np
from src.utils.logging import get_log_obj


def prepare_batch_obs(examples):
    return np.array(
        [
            np.concatenate(
                (
                    e["observation"]["obs"]["state"],
                    e["observation"]["obs"]["goal"],
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


class BitFlipRulePredictorSkeleton:

    def __init__(self, game, args):
        self.game = game
        self.net = BitFlipNNet(game=game, args=args)
        self.args = args
        self.logger = get_log_obj(args=args, name="AlphaZeroRulePredictor")

    def train(self, examples: typing.List):
        """
        This function trains the neural network with data gathered from self-play.
        :param examples: a list of training examples of the form: (o_t, (pi_t, v_t), w_t)
        """
        obs, target_pis, target_vs = prepare_batch(examples)
        # target_vs = self.scale_returns(target_vs)

        metrics = self.net.model.fit(
            x=obs,
            y=[target_pis, target_vs],
            batch_size=len(obs),
            epochs=1,
            verbose=False,
        )
        return (
            metrics.history["pi_loss"][0],
            metrics.history["v_loss"][0],
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

    # def predict_with_loss(self, examples):
    #     obs, pi, z = prepare_batch(examples)
    #     return self.net.model.evaluate(x=obs, y=[pi, z])

    # def scale_returns(self, zs):
    #     return np.array([z / self.game.getActionSize() for z in zs])
