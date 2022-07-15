import numpy as np


def load_data(DATA_DIR: str, validation_split: float, test_split: float):
    """Loads data from the /testing_data/ folder.

    Parameters:
    DATA_DIR (str):
    validation_split (float):
    test_split (float):

    Returns:
    x_train (numpy.array()): array of training input data
    y_train (numpy.array()): array of training output data
    """

    input = np.load(DATA_DIR + "x_pulse_AT.npy")
    output = np.load(DATA_DIR + "y_AT.npy")

    print(f"Input data shape: {input.shape}")
    print(f"Output data shape: {output.shape}")

    np.random.shuffle(input)
    np.random.shuffle(output)

    numel = input.shape[0]
    num_val = int(np.floor(numel * validation_split))
    num_test = int(np.floor(numel * test_split))
    num_train = int(numel - num_val - num_test)

    x_train = input[:num_train, :, :]
    x_val = input[num_train:num_val, :, :]
    x_test = input[-num_train:, :, :]

    y_train = output[:num_train, :, :]
    y_val = output[num_train:num_val, :, :]
    y_test = output[-num_train:, :, :]

    return x_train, y_train, x_val, y_val, x_test, y_test
