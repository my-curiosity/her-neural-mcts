import tensorflow as tf
import typing

from src.neural_nets.get_rule_predictor_class import get_rule_predictor_bitflip
import numpy as np
import src.neural_nets.loss as loss
from src.utils.logging import get_log_obj
from src.utils.tensors import check_for_non_numeric_and_replace_by_0
from src.utils.tensors import tf_save_cast_to_float_32


class BitFlipRulePredictorSkeleton(tf.keras.Model):

    def __init__(self, args):
        super(BitFlipRulePredictorSkeleton, self).__init__()
        self.single_player = (True,)

        self.net = get_rule_predictor_bitflip(args=args)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        self.args = args

        self.training = None
        self.steps = 0
        self.logger = get_log_obj(args=args, name="AlphaZeroRulePredictor")

    def train(self, examples: typing.List):
        """
        This function trains the neural network with data gathered from self-play.

        :param examples: a list of training examples of the form: (o_t, (pi_t, v_t), w_t)
        """
        self.net.training = True
        representation, target_pis, target_vs = self.prepare_batch_for_nn(examples)
        pi_batch_loss, v_batch_loss = self.train_step(
            representation=representation,
            target_pis=target_pis,
            target_vs=target_vs,
        )

        self.steps += 1
        return pi_batch_loss, v_batch_loss, 0

    def prepare_batch_for_nn(self, examples):
        observations, loss_scale, target_pis, target_vs = [], [], [], []
        for example in examples:
            observations.append(example["observation"]["obs"]["state"])
            if "probabilities_actor" in example:
                target_pis.append(example["probabilities_actor"])
                target_vs.append(example["observed_return"])
                loss_scale.append(example["loss_scale"])

        target_pis = tf_save_cast_to_float_32(
            x=target_pis, logger=self.logger, name="target_pis"
        )
        target_vs = tf_save_cast_to_float_32(
            x=target_vs, logger=self.logger, name="target_vs"
        )

        return (
            observations,
            target_pis,
            target_vs,
        )

    @tf.function
    def train_step(
        self,
        representation,
        target_pis,
        target_vs,
    ):
        with tf.GradientTape(persistent=True) as tape:
            action_prediction, v = self.net(nnet_input=representation)

            pi_batch_loss = loss.kl_divergence(real=target_pis, pred=action_prediction)
            v_batch_loss = loss.mean_square_error_loss_function(real=target_vs, pred=v)

        variables = [
            resourceVariable for resourceVariable in self.net.trainable_variables
        ]
        gradients = tape.gradient(pi_batch_loss, variables)
        gradients = [
            check_for_non_numeric_and_replace_by_0(
                logger=self.logger, tensor=x, name="target_pis"
            )
            for x in gradients
        ]
        self.optimizer.apply_gradients(zip(gradients, variables))

        return pi_batch_loss, v_batch_loss

    def predict(self, examples):
        action_prediction, v, _, _, _ = self.predict_with_loss(
            examples, with_loss=False
        )
        return action_prediction, v

    def predict_with_loss(self, examples, with_loss=True):
        self.net.training = False
        representation, target_pis, target_vs = self.prepare_batch_for_nn(examples)
        action_prediction, v = self.net(nnet_input=representation)
        if with_loss:
            pi_batch_loss = loss.kl_divergence(real=target_pis, pred=action_prediction)
            v_batch_loss = loss.mean_square_error_loss_function(real=target_vs, pred=v)
        else:
            pi_batch_loss = None
            v_batch_loss = None

        action_prediction = tf.squeeze(action_prediction).numpy().astype(np.float32)
        v = tf.squeeze(v).numpy().astype(np.float32)

        return action_prediction, v, pi_batch_loss, v_batch_loss, 0
