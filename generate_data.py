#!/usr/bin/env python3

import numpy as np
import pynlo
import time
import csv
from tqdm import tqdm


def main():
    # pulse parameters
    fwhm_list = np.linspace(start=0.100, stop=0.200, num=64)
    wl = 1030  # pulse central wavelength (nm)
    epp_list = np.linspace(start=10e-12, stop=100e-12, num=64)
    gdd = 0.0  # group delay dispersion (ps^2)
    tod = 0.0  # third order dispersion (ps^3)
    frep = 1  # repetition rate (MHz)

    # simulation parameters
    window = 10.0  # simulation window (ps)
    steps = 100  # simulation steps
    points = 2**13  # simulation points

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

    # run simulation for each parameter
    for fwhm in tqdm(fwhm_list):
        for epp in epp_list:
            # create file
            path_dir = "testing_data/"
            filename = f"nlse_fwhm-{fwhm*1e3:.2f}fs_epp-{epp*1e9:.3e}nJ_time-{int(time.time())}"
            data = {}

            # create pulse
            pulse = pynlo.light.DerivedPulses.SechPulse(
                power=1,  # power will be scaled by set_epp()
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

            # save data
            data["pulse_AW"] = pulse.AW
            data["pulse_AT"] = pulse.AT
            data["fwhm"] = fwhm
            data["epp"] = epp
            data["AW"] = AW
            data["AT"] = AT

            with open(path_dir + filename + ".csv", "w") as f:
                w = csv.DictWriter(f, data.keys())
                w.writeheader()
                w.writerow(data)


if __name__ == "__main__":
    main()
