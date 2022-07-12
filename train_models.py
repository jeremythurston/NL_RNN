#!/usr/bin/env python3

import time
import build_model
import load_data
import pynlo
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from kerastuner.tuners import RandomSearch


def train_models():
    """Trains models.

    Returns the best one.
    """
    validation_split = 0.2
    epochs = 10
    batch_size = 32

    DATA_DIR = (
        "/nlse__1024sims__fwhm-100.0-200.0fs__epp-1.00e-02-1.00e-01nJ__time-1657572657/"
    )
    LOG_DIR = "logs/"

    x_train, y_train = load_data.load_data(DATA_DIR)

    # Generate log file for callbacks
    # NAME = f"NL_RNN_lr-{lr}_window-{window}_grusize-{gru_size}_densesize-{dense_size}_time-{int(time.time())}"
    # tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

    # Build model
    # model = build_model.build_model(
    #     pts=pts, lr=lr, window=window, gru_size=gru_size, dense_size=dense_size
    # )

    # Reduce learning rate on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor="val_mse", factor=0.5, patience=3, min_lr=1e-6
    )

    # Use tuner to find best model
    tuner = RandomSearch(
        build_model.build_model,
        objective="val_mse",
        max_trials=5,
        executions_per_trial=2,
        directory=LOG_DIR,
    )

    tuner.search(
        x=x_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[TensorBoard(LOG_DIR), reduce_lr],
    )

    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters()[0]

    # Create model
    model = tuner.hypermodel.build(best_hps)

    # Make prediction with best model
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=[TensorBoard(LOG_DIR)],
    )

    plt.figure()

    plt.plot(history.history["val_mse"])
    plt.xlabel("Epochs")
    plt.ylabel("Validation MSE")
    plt.yscale("log")

    plt.show()


def make_prediction(model):
    plt.style.use(["science", "nature", "bright"])

    # Parameters
    fwhm = 0.18
    wl = 1030
    window = 10.0
    gdd = 0
    tod = 0
    points = 2**10
    frep = 1
    epp = 50e-12

    pulse = pynlo.light.DerivedPulses.SechPulse(
        power=1,
        T0_ps=fwhm / 1.76,
        center_wavelength_nm=wl,
        time_window_ps=window,
        GDD=gdd,
        TOD=tod,
        NPTS=points,
        frep_MHz=frep,
        power_is_avg=False,
    )
    pulse.set_epp(epp)

    pulse_profile = pulse.AT

    prediction = model.predict(pulse_profile)

    plt.imshow(prediction)
    plt.show()


if __name__ == "__main__":
    train_models()
