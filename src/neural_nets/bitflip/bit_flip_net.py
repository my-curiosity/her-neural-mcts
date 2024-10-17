import tensorflow as tf


class BitFlipNNet(tf.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.num_bits = kwargs["num_bits"]
        self.name_of_net = kwargs["name"]
        self.args = args

        self.dense_1 = tf.keras.layers.Dense(
            units=20,
            activation="relu",
            dtype=tf.float32,
            name=f"{kwargs['name']}_Dense_1",
        )
        self.pi = tf.keras.layers.Dense(
            units=self.num_bits,
            activation="softmax",
            dtype=tf.float32,
            name=f"{kwargs['name']}_pi",
        )
        self.v = tf.keras.layers.Dense(
            units=1, activation="tanh", dtype=tf.float32, name=f"{kwargs['name']}_v"
        )

    def __call__(self, x, training):

        x = tf.expand_dims(
            x,
            axis=0,
        )
        x = self.dense_1(x)
        pi = self.pi(x)
        v = self.v(x)

        return pi, v

    def __str__(self):
        return f"BitFlip_MLP_{self.name_of_net}"
