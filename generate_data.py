#!/usr/bin/env python3

import numpy as np
import pynlo
import time
import os
from tqdm import tqdm


def main():
    # pulse parameters
    fwhm_list = np.linspace(start=0.100, stop=0.200, num=32)
    wl = 1030  # pulse central wavelength (nm)
    epp_list = np.linspace(start=10e-12, stop=100e-12, num=32)
    gdd = 0.0  # group delay dispersion (ps^2)
    tod = 0.0  # third order dispersion (ps^3)
    frep = 1  # repetition rate (MHz)

    # simulation parameters
    window = 10.0  # simulation window (ps)
    steps = 100  # simulation steps
    points = 2**11  # simulation points
    batch_size = fwhm_list.size * epp_list.size  # number of total simulations
    rnn_window = 10

    # fiber parameters
    beta2 = -120  # (ps^2/km)
    beta3 = 0.00  # (ps^3/km)
    beta4 = 0.005  # (ps^4/km)
    length = 50  # length in mm
    alpha = 0.0  # attentuation coefficient (dB/cm)
    alpha = np.log((10 ** (alpha * 0.1))) * 100  # convert from dB/cm to 1/m
    gamma = 1000  # gamma (1/(W km)
    fib_wl = 1064  # central wl of fiber (nm)

    # NLSE terms
    raman = True  # enable Raman effect?
    steep = True  # enable self steepening?

    # generate arrays to store data
    x_pulse_AW = np.zeros((batch_size, points), dtype=complex)
    x_pulse_AT = np.zeros((batch_size, points), dtype=complex)
    y_AW = np.zeros((batch_size, points, steps), dtype=complex)
    y_AT = np.zeros((batch_size, points, steps), dtype=complex)

    pathdir = f"./testing_data/nlse__{batch_size}sims__fwhm-{np.min(fwhm_list)*1e3:.1f}-{np.max(fwhm_list)*1e3:.1f}fs__epp-{np.min(epp_list)*1e9:.2e}-{np.max(epp_list)*1e9:.2e}nJ__time-{int(time.time())}/"
    os.makedirs(pathdir)

    # run simulation for each parameter
    for i, fwhm in enumerate(tqdm(fwhm_list)):
        for j, epp in enumerate(epp_list):
            idx = int(i * fwhm_list.size / 32 + j * epp_list.size / 32 - 1)

            # create pulse
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

            # create fiber
            fiber = pynlo.media.fibers.fiber.FiberInstance()
            fiber.generate_fiber(
                length * 1e-3,
                center_wl_nm=fib_wl,
                betas=(beta2, beta3, beta4),
                gamma_W_m=gamma * 1e-3,
                gvd_units="ps^n/km",
                gain=-alpha,
            )

            # propagate
            evol = pynlo.interactions.FourWaveMixing.SSFM.SSFM(
                local_error=0.005,
                USE_SIMPLE_RAMAN=True,
                disable_Raman=np.logical_not(raman),
                disable_self_steepening=np.logical_not(steep),
            )
            _, AW, AT, _ = evol.propagate(pulse_in=pulse, fiber=fiber, n_steps=steps)

            x_pulse_AW[idx, :] = pulse.AW
            x_pulse_AT[idx, :] = pulse.AT
            y_AW[idx, :, :] = AW
            y_AT[idx, :, :] = AT

    # normalize data
    x_pulse_AW = x_pulse_AW / np.max(np.absolute(x_pulse_AW))
    x_pulse_AT = x_pulse_AT / np.max(np.absolute(x_pulse_AT))
    y_AW = y_AW / np.max(np.absolute(y_AW))
    y_AT = y_AT / np.max(np.absolute(y_AT))

    # duplicate first row (rnn_window) times
    x_pulse_AW = np.repeat(x_pulse_AW[:, np.newaxis, :], rnn_window, axis=1)
    x_pulse_AT = np.repeat(x_pulse_AT[:, np.newaxis, :], rnn_window, axis=1)

    # surely there's a faster way to do this
    temp = y_AW[:, :, 0]
    temp2 = np.repeat(temp[:, :, np.newaxis], rnn_window, axis=2)
    y_AW = np.append(temp2, y_AW, axis=2)
    y_AW = np.swapaxes(y_AW, 1, 2)
    temp = y_AT[:, :, 0]
    temp2 = np.repeat(temp[:, :, np.newaxis], rnn_window, axis=2)
    y_AT = np.append(temp2, y_AT, axis=2)
    y_AT = np.swapaxes(y_AT, 1, 2)

    # save data
    np.save(pathdir + "x_pulse_AW.npy", x_pulse_AW)
    np.save(pathdir + "x_pulse_AT.npy", x_pulse_AT)
    np.save(pathdir + "y_AW.npy", y_AW)
    np.save(pathdir + "y_AT.npy", y_AT)


if __name__ == "__main__":
    main()
