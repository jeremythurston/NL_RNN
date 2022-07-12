#!/usr/bin/env python3

import glob
import matplotlib.pyplot as plt
import pandas as pd


def main():
    plt.style.use(["science", "nature", "bright"])

    files = glob.glob("./loss_data/*.csv")

    plt.figure()

    for file in files:
        df = pd.read_csv(file)
        epoch = df["Step"]
        loss = df["Value"]

        plt.plot(epoch, loss)

    plt.xlabel("Epoch")
    plt.ylabel("Validation MSE")
    plt.yscale("log")

    plt.savefig("lossplot.png", dpi=1000)


if __name__ == "__main__":
    main()
