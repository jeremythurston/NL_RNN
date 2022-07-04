from tensorflow import keras
from keras import layers, optimizers, losses
from kerastuner.engine.hyperparameters import HyperParameters


def build_model(hp) -> keras.Model():
    """Builds the neural network.

    Parameters:
    pts (integer): number of grid points in simulation
    lr (float): learning rate
    window (int): number of simulation steps used to predict the next step
    gru_size (int): nodes in the GRU layer
    dense_size (int): nodes in the dense layer

    Returns:
    model (keras.Model()): the neural network model class
    """

    window = hp.Int("window_size", min_value=5, max_value=20)
    pts = 2**13

    model = keras.Sequential()

    model.add(
        layers.GRU(
            hp.Int("gru_nodes", min_value=64, max_value=256, step=64), activation="relu", input_shape=(window, pts)
        )
    )
    for i in range(hp.Int("dense_layers", min_value=0, max_value=4)):
        model.add(
            layers.Dense(
                hp.Int(f"dense_nodes_{i}", min_value=64, max_value=256, step=64),
                activation="relu",
            )
        )
    model.add(layers.Dense(2**13, activation="sigmoid"))

    optimizer = optimizers.Adam(
        hp.Float("lr", min_value=1e-6, max_value=1e-3, sampling="log")
    )
    loss = losses.MeanSquaredError()

    model.compile(optimizer=optimizer, loss=loss, metrics=["mse"])

    return model
