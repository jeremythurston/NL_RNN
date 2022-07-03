from gc import callbacks
import time
import build_model
from keras.callbacks import TensorBoard


def main():
    # Hyperparameters
    lr = 1e-4
    window = 10
    epochs = 20
    validation_split = 0.333
    pts = 256
    gru_size = 256
    dense_size = 256

    # Generate filename for callback
    NAME = f"NL_RNN_lr-{lr}_window-{window}_grusize-{gru_size}_densesize-{dense_size}_time-{int(time.time())}"
    tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

    # Build model
    model = build_model.build_model(
        pts=pts, lr=lr, window=window, gru_size=gru_size, dense_size=dense_size
    )

    model.fit(
        train_in,
        train_out,
        validation_split=validation_split,
        epochs=epochs,
        callbacks=[tensorboard],
    )


if __name__ == "__main__":
    main()
