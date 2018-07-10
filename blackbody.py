# -*- coding: utf-8 -*-
#
#   blackbody.py
#   Author: Max Briel
#
#   Functions related to the blackbody emission and IC of its seed field
#

import numpy as np
from astropy import constants as const
from astropy import units as u
from helper_functions import scatter_absorb, get_bin
from collections import Counter
from random import random
from IC import upscatter


def BB_generator(T):
    """
    Return a function for a blackbody photon number density

    Input
    - T:            Blackbody temperatures  (keV)

    Output
    - function:     gives the number of photons for a given energy
    """
    # Set constants to the right value
    f1 = ((2.) / (const.h**3 * const.c**2)).to("1/(cm2*keV3*s)").value
    f2 = 1./ T
    def nr_photons(E):
        """
        Given an energy return the number of photons per (cm2 keV s)
        """
        E = E
        if f2 * E > 700:
            return (f1*(E**2))*np.exp(-f2*E)
        else:
            return (f1*(E**2))/(np.exp(f2*E)-1)
    return nr_photons


def do_BB_comp(nphotons, e_bins, TBB, tau, ratio, rnd_phE, rnd_electron):
    """
    Do emission and IC scattering of a blackbody photon seed field

    Input
    - nphotons:     Number of photons
    - nbins:        Number of bins
    - TBB:          Blackbody Temp          (keV)
    - tau:          optical depth
    - ratio:        normaisation factor
    - rnd_phE:      random E function
    - rnd_electron: random electron function

    Ouput
    - bin_dist:     number of photons per bin (#photons/(cm2*s*keV)  [Counter]
    """
    nr_photons = BB_generator(TBB)      #nr_photons(energy (keV))

    bin_dist = Counter({i:0 for i in range(len(e_bins)-1)})
    for i in range(nphotons):
        Eph = rnd_phE()
        Nph = nr_photons(Eph)
        prob = 1./(Eph*ratio)
        weight = Nph*prob
        while weight > 1e-6:
            # store escaping photons
            escape = weight*np.exp(-tau)
            ph_bin = get_bin(Eph, e_bins)
            bin_dist[ph_bin] += escape*Eph

            if scatter_absorb(Eph):
                gamma_electron = rnd_electron()
                theta = random()*2*np.pi
                Eph = upscatter(Eph, theta, gamma_electron)
                weight = weight*(1-np.exp(-tau))
            else:
                break

    return bin_dist
