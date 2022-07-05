#!/usr/bin/env python3

import time
import build_model
import load_data
from tensorflow import keras
from keras.callbacks import TensorBoard
from kerastuner.tuners import RandomSearch


def main():
    validation_split = 0.333
    epochs = 10
    batch_size = 64
    data_folder = "/nlse__1024sims__fwhm-100.0-200.0fs__epp-1.00e-02-1.00e-01nJ__time-1656998467/"
    LOG_DIR = "logs/"

    x_train, y_train = load_data.load_data(data_folder)

    # Generate log file for callbacks
    # NAME = f"NL_RNN_lr-{lr}_window-{window}_grusize-{gru_size}_densesize-{dense_size}_time-{int(time.time())}"
    # tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

    # Build model
    # model = build_model.build_model(
    #     pts=pts, lr=lr, window=window, gru_size=gru_size, dense_size=dense_size
    # )

    # Use tuner to find best model
    tuner = RandomSearch(
        build_model.build_model,
        objective="val_accuracy",
        max_trials=64,
        executions_per_trial=3,
        directory=LOG_DIR,
    )

    tuner.search(
        x=x_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[TensorBoard(LOG_DIR)],
    )

    # model.fit(
    #     train_in,
    #     train_out,
    #     validation_split=validation_split,
    #     epochs=epochs,
    #     callbacks=[tensorboard],
    # )


if __name__ == "__main__":
    main()
