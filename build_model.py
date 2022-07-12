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
    pts = 2**10
    steps = 100

    model = keras.Sequential()

    model.add(
        layers.GRU(
            hp.Int("gru_nodes", min_value=128, max_value=1152, step=512),
            activation="relu",
            input_shape=(window, pts),
        )
    )
    for i in range(hp.Int("dense_layers", min_value=5, max_value=8)):
        model.add(layers.Dropout(0.5))
        model.add(
            layers.Dense(
                hp.Int(f"dense_nodes_{i+1}", min_value=128, max_value=1152, step=512),
                activation="relu",
            )
        )
    model.add(layers.Dense(pts * (steps + window), activation="sigmoid"))
    model.add(layers.Reshape((steps + window, pts)))

    optimizer = optimizers.Adam(1e-4)
    loss = losses.MeanSquaredError()

    model.compile(optimizer=optimizer, loss=loss, metrics=["mse", "mae"])

    return model
