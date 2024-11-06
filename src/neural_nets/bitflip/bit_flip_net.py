from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    Input,
)
from tensorflow.keras.optimizers import Adam


class BitFlipNNet:
    def create_model(self):
        inputs = Input(shape=(2 * self.action_size))

        fc1 = Dropout(0.3)(
            Activation("relu")(BatchNormalization(axis=1)(Dense(1024)(inputs)))
        )
        fc2 = Dropout(0.3)(
            Activation("relu")(BatchNormalization(axis=1)(Dense(1024)(fc1)))
        )
        fc3 = Dropout(0.3)(
            Activation("relu")(BatchNormalization(axis=1)(Dense(512)(fc2)))
        )

        pi = Dense(self.action_size, activation="softmax", name="pi")(fc3)
        v = Dense(1, name="v")(fc3)  # TODO: no tanh?

        return Model(inputs=inputs, outputs=[pi, v])

    def __init__(self, game, args):
        self.action_size = game.getActionSize()
        self.args = args

        self.model = self.create_model()
        self.model.compile(
            loss=["categorical_crossentropy", "mean_squared_error"],
            optimizer=Adam(learning_rate=0.0005),
        )
