import time
import build_model
from tensorflow import keras
from keras.callbacks import TensorBoard
from kerastuner.tuners import RandomSearch


def main():
    validation_split = 0.333

    # Generate log file for callbacks
    # NAME = f"NL_RNN_lr-{lr}_window-{window}_grusize-{gru_size}_densesize-{dense_size}_time-{int(time.time())}"
    # tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

    LOG_DIR = "logs/"

    # Build model
    # model = build_model.build_model(
    #     pts=pts, lr=lr, window=window, gru_size=gru_size, dense_size=dense_size
    # )

    # Use tuner to find best model
    tuner = RandomSearch(
        build_model.build_model,
        objective="val_accuracy",
        max_trials=3,
        executions_per_trial=3,
        directory=LOG_DIR,
    )

    tuner.search(
        x=x_train,
        y=y_train,
        epochs=10,
        batch_size=64,
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
