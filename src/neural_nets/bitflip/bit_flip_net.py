from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Input,
)
from tensorflow.keras.optimizers import Adam


class BitFlipNNet:
    def create_model(self):
        inputs = Input(shape=(2 * self.num_bits))

        shared = Activation("relu")((Dense(20)(inputs)))
        pi = Dense(self.action_size, activation="softmax", name="pi")(shared)
        v = Dense(1, activation="tanh", name="v")(shared)

        return Model(inputs=inputs, outputs=[pi, v])

    def __init__(self, game, args):
        self.num_bits = self.action_size = game.getActionSize()
        self.args = args

        self.model = self.create_model()
        self.model.compile(
            loss=["categorical_crossentropy", "mean_squared_error"],
            optimizer=Adam(learning_rate=0.0005),
        )
