#!/usr/bin/env python3

import pynlo
import train_models
from tensorflow import keras
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    make_prediction(model)
