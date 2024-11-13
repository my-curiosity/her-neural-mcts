from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    Input,
)
from tensorflow.keras.optimizers import Adam


class MazeNNet:
    def create_model(self):
        inputs = Input(shape=self.observation_size)

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
        v = Dense(1, activation="linear", name="v")(fc3)

        return Model(inputs=inputs, outputs=[pi, v])

    def __init__(self, game, args):
        self.observation_size = 4 + 2
        self.action_size = game.getActionSize()
        self.args = args

        self.model = self.create_model()
        self.model.compile(
            loss=["categorical_crossentropy", "mean_squared_error"],
            optimizer=Adam(learning_rate=0.001),
        )
