import numpy as np


def load_data(filename: str):
    """
    Parameters:
    filename (string): relative filename of numpy data

    Returns:
    pts (integer): number of grid points in simulation
    train_in (numpy.array()): array of training input data
    train_out (numpy.array()): array of training output data
    test_in (numpy.array()): array of testing input data
    test_out (numpy.array()): array of testing output data
    """

    data = np.loadtxt(filename, dtype=float)

    return pts, train_in, train_out, test_in, test_out
