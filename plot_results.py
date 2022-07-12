from keras.callbacks import History
import matplotlib.pyplot as plt


def plot_results(history: History()) -> None:
    """Plots the results of a trained model.

    Parameters:
    history (History()): the history of training a model
    """

    plt.style.use(["science", "nature", "bright"])

    plt.figure()

    plt.plot(history.history["val_mse"])
    plt.xlabel("Epochs")
    plt.ylabel("Validation MSE")
    plt.yscale("log")

    plt.show()
