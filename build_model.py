from tensorflow import keras
from keras import layers, optimizers, losses


def build_model(
    pts: int, lr: float, window: int, gru_size: int, dense_size: int
) -> keras.Model():
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

    model = keras.Sequential()

    model.add(layers.GRU(gru_size, activation="relu"))
    model.add(layers.Dense(dense_size, activation="relu"))
    model.add(layers.Dense(dense_size, activation="relu"))
    model.add(layers.Dense(pts, activation="sigmoid"))

    model.summary()

    optimizer = optimizers.Adam(lr)
    loss = losses.MeanSquaredError()

    model.compile(optimizer=optimizer, loss=loss, metrics=["mse"])

    return model
