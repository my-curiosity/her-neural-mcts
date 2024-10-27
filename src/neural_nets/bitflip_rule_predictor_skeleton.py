import typing

from src.neural_nets.bitflip.bit_flip_net import BitFlipNNet
import numpy as np
from src.utils.logging import get_log_obj


def prepare_for_prediction(examples):
    return np.expand_dims(
        np.concatenate(
            (
                examples[0]["observation"]["obs"]["state"],
                examples[0]["observation"]["obs"]["goal"],
            )
        ),
        axis=0,
    )


def prepare_batch_for_training(examples):
    observations, loss_scale, target_pis, target_vs = [], [], [], []
    for example in examples:
        observations.append(
            np.concatenate(
                (
                    example["observation"]["obs"]["state"],
                    example["observation"]["obs"]["goal"],
                )
            )
        )
        if "probabilities_actor" in example:
            target_pis.append(example["probabilities_actor"])
            target_vs.append(example["observed_return"])
            loss_scale.append(example["loss_scale"])
    return (
        np.asarray(observations),
        np.asarray(target_pis),
        np.asarray(target_vs),
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
        inputs, target_pis, target_vs = prepare_batch_for_training(examples)
        metrics = self.net.model.fit(
            x=inputs,
            y=[target_pis, target_vs],
            batch_size=self.args.batch_size_training,
            epochs=1,
        )
        return metrics.history["pi_loss"][0], metrics.history["v_loss"][0], 0

    def predict(self, examples):
        inputs = prepare_for_prediction(examples)
        pi, v = self.net.model.predict(inputs, verbose=False)
        return pi[0], v[0][0]
