# -*- coding: utf-8 -*-
#
#   synchrotron.py
#   Author: Max Briel
#
#   Functions to calculate the synchrotron spectrum and generate
#   a number of photons function

import astropy.constants as const
import astropy.units as u
import numpy as np
from helper_functions import scatter_absorb, get_bin
from random import random
from collections import Counter
from IC import upscatter

def nrph_sync_generator(emissivity, radius):
    """
    Generates a photon flux function
    """
    def nrph_sync(energy):
        return ((1./6) *
                emissivity(energy)*radius)/(energy*const.h.to(u.erg/u.Hz).value)

    return nrph_sync

def do_synchrotron(nphotons, e_bins, radius, tau, ratio,
                   emissivity, rnd_phE, rnd_electron):
    """
    Do synchrtron emission and Inverse Compton uscattering.

    Input
    - nphotons:     Number of photons
    - nbins:        Number of bins
    - TBB:          Blackbody Temp          (keV)
    - tau:          optical depth
    - ratio:        normaisation factor
    - rnd_phE:      random E function
    - rnd_electron: random electron function

    Ouput
    - bin_dist:     number of photons per bin (#photons/(cm2*s*keV) [Counter]
    """
    nr_photons = nrph_sync_generator(emissivity, radius)
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
