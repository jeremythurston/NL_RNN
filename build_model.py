from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses

def build_model(pts: int, lr: float, window: int):
    """
    Parameters:
    pts (integer): number of grid points in simulation
    lr (float): learning rate
    window (int): number of simulation steps used to predict the next step

    Returns:
    model (keras.Model()): the neural network model class
    """

    model = keras.Sequential()

    model.add(layers.GRU(256, activation="relu"))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(pts, activation="sigmoid"))

    optimizer = optimizers.Adam(lr)
    loss = losses.MeanSquaredError()

    model.compile(optimizer=optimizer, loss=loss, metrics=["mse"])

    return model