import tensorflow as tf
from src.utils.logging import get_log_obj


class BitFlipRulePredictorNet(tf.keras.Model):

    def __init__(
        self,
        nnet_class,
        nnet_args,
        args,
    ):
        super(BitFlipRulePredictorNet, self).__init__()
        self.nnet = nnet_class(**nnet_args)

        self.training = None
        self.args = args
        self.logger_net = get_log_obj(args=args, name="RulePredictorNet")

    @tf.function
    def __call__(self, nnet_input):
        action, value = self.nnet(x=nnet_input, training=self.training)
        return action, value
