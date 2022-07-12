import numpy as np


def load_data(data_folder: str):
    """Loads data from the /testing_data/ folder.

    Returns:
    x_train (numpy.array()): array of training input data
    y_train (numpy.array()): array of training output data
    """

    data_dir = "./testing_data" + data_folder

    x_train = np.load(data_dir + "x_pulse_AT.npy")
    y_train = np.load(data_dir + "y_AT.npy")

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    return x_train, y_train
