from keras.callbacks import History
import matplotlib.pyplot as plt


def plot_results(history: History()) -> None:
    """Plots the results of a trained model.

    Parameters:
    history (History()): the history of training a model

    Returrns:
    None
    """

    plt.style.use(["science", "nature", "bright"])

    # plt.figure()

    # plt.plot(history.history["mse"], label="Training")
    # plt.plot(history.history["val_mse"], label="Validation")

    # plt.xlabel("Epochs")
    # plt.ylabel("MSE")
    # plt.yscale("log")
    # plt.legend()

    # plt.show()

    # test_pulse, freqs = get_pulse()
    # prediction = model.predict(test_pulse)
    # ev = freqs * 0.004136  # THz to eV

    # plt.style.use(["science", "nature", "bright"])

    # _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    # ax1.plot(ev, np.absolute(np.transpose(test_pulse[0, :])))
    # ax1.set_xlabel("Photon energy (eV)")
    # ax1.set_title("Prediction input")

    # ax2.imshow(prediction[0, :, :], aspect="auto", cmap="jet")
    # ax2.set_title("Prediction output")

    # ax3.plot(x_train[0, 0, :] / np.max(x_train[0, 0, :]))
    # ax3.set_title("Train input")

    # ax4.imshow(np.absolute(y_train[0, :, :]), aspect="auto", cmap="jet")
    # ax4.set_title("Train output")

    # plt.show()
