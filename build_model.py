import generate_data
from tensorflow import keras
from keras import layers, optimizers, losses
from kerastuner.engine.hyperparameters import HyperParameters


def build_model(hp) -> keras.Model():
    """Builds the neural network. Uses KerasTuner to iterate over hyperparameters.

    Returns:
    model (keras.Model()): the neural network model class
    """

    # TODO: somehow get these from load_data or generate_data
    window = 10
    pts = 2**11

    model = keras.Sequential()

    model.add(
        layers.GRU(
            hp.Int("gru_nodes", min_value=64, max_value=256, step=64),
            activation="relu",
            input_shape=(window, pts),
        )
    )
    for i in range(hp.Int("dense_layers", min_value=0, max_value=4)):
        model.add(
            layers.Dense(
                hp.Int(f"dense_nodes_{i}", min_value=64, max_value=256, step=64),
                activation="relu",
            )
        )
    model.add(layers.Dense(pts, activation="sigmoid"))

    optimizer = optimizers.Adam(
        hp.Float("lr", min_value=1e-6, max_value=1e-3, sampling="log")
    )
    loss = losses.MeanSquaredError()

    model.compile(optimizer=optimizer, loss=loss, metrics=["mse"])

    return model
