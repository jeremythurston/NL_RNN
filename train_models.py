import build_model
import load_data
import plot_results
import pynlo
import matplotlib.pyplot as plt
import numpy as np
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

    x_train, y_train = load_data.load_data(DATA_DIR)

    # Reduce learning rate on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor="val_mse", factor=0.5, patience=3, min_lr=1e-6
    )

    # Use tuner to find best model
    tuner = RandomSearch(
        build_model.build_model,
        objective="val_mse",
        max_trials=1,
        executions_per_trial=1,
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

    # Train best model
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=[TensorBoard(LOG_DIR)],
    )

    plot_results.plot_results(history)

    # REFORMAT BELOW
    # --------------
    test_pulse, freqs = get_pulse()
    prediction = model.predict(test_pulse)
    ev = freqs * 0.004136  # THz to eV

    plt.style.use(["science", "nature", "bright"])

    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    ax1.plot(ev, np.absolute(np.transpose(test_pulse[0, :])))
    ax1.set_xlabel("Photon energy (eV)")
    ax1.set_title("Prediction input")

    ax2.imshow(prediction[0, :, :], aspect="auto", cmap="jet")
    ax2.set_title("Prediction output")

    ax3.plot(x_train[0, 0, :] / np.max(x_train[0, 0, :]))
    ax3.set_title("Train input")

    ax4.imshow(np.absolute(y_train[0, :, :]), aspect="auto", cmap="jet")
    ax4.set_title("Train output")

    plt.show()


def get_pulse():
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
    rnn_window = 10

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

    pulse_profile = pulse_profile / np.max(np.absolute(pulse_profile))
    pulse_profile = np.repeat(
        pulse_profile[np.newaxis, np.newaxis, :], rnn_window, axis=1
    )

    return pulse_profile, pulse.F_THz
