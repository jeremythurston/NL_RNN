import build_model
import load_data
import plot_results
import pynlo
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from kerastuner.tuners import RandomSearch


def train_models(
    DATA_DIR: str,
    LOG_DIR: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    validation_data: (np.ndarray, np.ndarray),
    epochs: int,
    batch_size: int,
):
    """Uses a KerasTuner to iterate over hyperparameters and returns the best one

    Parameters:
    DATA_DIR (str): Directory of the data to be trained on
    LOG_DIR (str): Directory of logs to be used for callbacks

    Returns:
    hps (HyperParameters()): Hyperparameters of the best performing model
    """

    # Moved to RNN_main.ipynb
    # x_train, y_train = load_data.load_data(DATA_DIR)

    # Reduce learning rate on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor="val_mse", factor=0.5, patience=3, min_lr=1e-6
    )

    # Use tuner to find best model
    # RandomSearch tuner
    tuner = RandomSearch(
        build_model.build_model,
        objective="val_mse",
        max_trials=1,
        executions_per_trial=1,
        directory=LOG_DIR,
    )
    # Bayesian Optimizer tuner
    # tuner = TODO

    tuner.search(
        x=x_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=[TensorBoard(LOG_DIR), reduce_lr],
    )

    # Get best hyperparameters
    hps = tuner.get_best_hyperparameters()[0]

    return tuner, hps

    # Moved below to train_best()
    # # Create model
    # model = tuner.hypermodel.build(best_hps)

    # # Train best model
    # history = model.fit(
    #     x=x_train,
    #     y=y_train,
    #     epochs=epochs,
    #     validation_split=validation_split,
    #     callbacks=[TensorBoard(LOG_DIR)],
    # )

    # plot_results.plot_results(history)

    # Moved to plot_results.py
    # --------------
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


def train_best(
    DATA_DIR: str,
    LOG_DIR: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    validation_data: (np.ndarray, np.ndarray),
    tuner,
    hps,
    epochs: int,
):
    """Takes the hyperparameters with the best performance, trains it, then returns the model.

    Call this function after train_models()

    Parameters:
    DATA_DIR (str): Directory of the data to be trained on
    LOG_DIR (str): Directory of logs to be used for callbacks
    tuner (RandomSearch()): Tuner object
    hps (HyperParameter()): Hyperparameters to be used for training
    epochs (int): Number of epochs to train
    validation_split (float): Fraction of data to be saved for validation during training

    Returns:
    model (Model()): Trained model
    history (History()): History of the best model's training
    """

    # Moved to RNN_main.ipynb
    # Get data
    # x_train, y_train = load_data.load_data(DATA_DIR)

    # Create model
    model = tuner.hypermodel.build(hps)

    # Train best model
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=[TensorBoard(LOG_DIR)],
    )

    # Plot training results
    plt.style.use(["science", "notebook", "bright"])

    plt.figure()

    plt.plot(history.history["mse"], label="Training")
    plt.plot(history.history["val_mse"], label="Validation")

    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.yscale("log")
    plt.legend()

    plt.show()

    return model


# Moved to RNN_main.ipynb
# def get_pulse():
#     plt.style.use(["science", "nature", "bright"])

#     # Parameters
#     fwhm = 0.18
#     wl = 1030
#     window = 10.0
#     gdd = 0
#     tod = 0
#     points = 2**10
#     frep = 1
#     epp = 50e-12
#     rnn_window = 10

#     pulse = pynlo.light.DerivedPulses.SechPulse(
#         power=1,
#         T0_ps=fwhm / 1.76,
#         center_wavelength_nm=wl,
#         time_window_ps=window,
#         GDD=gdd,
#         TOD=tod,
#         NPTS=points,
#         frep_MHz=frep,
#         power_is_avg=False,
#     )
#     pulse.set_epp(epp)

#     pulse_profile = pulse.AT

#     pulse_profile = pulse_profile / np.max(np.absolute(pulse_profile))
#     pulse_profile = np.repeat(
#         pulse_profile[np.newaxis, np.newaxis, :], rnn_window, axis=1
#     )

#     return pulse_profile, pulse.F_THz
