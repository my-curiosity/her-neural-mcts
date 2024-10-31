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
        inputs = Input(shape=(2 * self.action_size))

        fc1 = Activation("relu")(BatchNormalization(axis=1)(Dense(128)(inputs)))
        fc2 = Activation("relu")(BatchNormalization(axis=1)(Dense(128)(fc1)))
        fc3 = Activation("relu")(BatchNormalization(axis=1)(Dense(64)(fc2)))

        pi = Dense(self.action_size, activation="softmax", name="pi")(fc3)
        v = Dense(1, activation="tanh", name="v")(fc3)

        return Model(inputs=inputs, outputs=[pi, v])

    def __init__(self, game, args):
        self.action_size = game.getActionSize()
        self.args = args

        self.model = self.create_model()
        self.model.compile(
            loss=["categorical_crossentropy", "mean_squared_error"],
            optimizer=Adam(learning_rate=0.0005),
        )
