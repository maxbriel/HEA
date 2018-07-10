# -*- coding: utf-8 -*-
#
#   helper_functions.py
#   Author: Max Briel
#
#   Helper functions to aid the calculations in the main Monte Carlo
#   simulation and to provide useful functions needed in
#   synchrotron and blackbody spectra calculations
#


import numpy as np
import astropy.units as u
import astropy.constants as const
from random import random

# ---------------------------------------------------------------------------
#                           FUNCTIONS TO CALCULATE PHYSICAL PROPERTIES
# ---------------------------------------------------------------------------
def magnetic_field(TE, n_e):
    """
    Calculate the magnetic field strenght in Gauss using the electron
    temperature and electron density.

    Input
    - TE:   electron temperature    (keV)
    - n_e:  electron density        (#electrons/cm3)

    Ouput:
    - B:    magnetic Field          (Gauss)
    """
    B = np.sqrt((24*np.pi*TE*u.keV*n_e/
                                (u.cm**3))).decompose(bases=u.cgs.bases).value
    return B

def electron_density(radius, TE, P_corona):
    """
    Gives the electron density given the radius, electron temperature and
    the corona's power output.

    Input
    - radius:   radius of the source  (cm)
    - TE:       electron temperature  (K)
    - P_corona: Corona power          (keV/K)

    Ouput
    - n_e:      electron density      (1/cm3)
    """
    c = const.c.to("cm/s").value
    return P_corona / (12*np.pi*(radius**2)*c*TE)


def grav_radius(mass):
    """
    Given the mass in solar mass units, returns the gravitational radius in cm

    Input
    - Mass:     Mass of object  (solmass)

    Ouput
    - Gravitational Radius:     (cm)
    """
    mass = mass * u.solMass
    return (const.G*mass / const.c**2).to(u.cm).value


def electron_number(n_e, pdf_e):
    """
    Returns a function that that gives the electron
    density at an electron gamma.

    Input
    - n_e:      electron density                (#electrons/cm3)
    - pdf_e:    Probability density function

    Ouput
    - function: Takes a electron gamma and returns the electron density
    """
    return lambda g: n_e*pdf_e(g)


# ---------------------------------------------------------------------------
#                   FUNCTIONS FOR THE MONTE CARLO SIMULATION
# ---------------------------------------------------------------------------

def pdf_min_max(pdf_e, e_bins):
    """
    Given a probability density function and an energy range return the maximum
    x and y value above 1e-10.

    Input
    - pdf_e: probability density function of electron
    - e_bins: bins of energies

    Output
    - xmax: maximum x value where pdf > 1e-10
    - ymax: maximum y value of the pdf
    """
    gammas = np.logspace(0,10,10000)
    prob = np.array([pdf_e(i) for i in gammas])
    ymax = np.max(prob)
    xmax_index = np.flatnonzero(prob > 1e-10)[-1]
    xmax = gammas[xmax_index]
    return (xmax, ymax)

def get_bin(energy, e_bins):
    """
    Get the bin number of the energy

    Input
    - energy
    - e_bins:   Array of starting values for e_bins

    Output:
    - bin number
    """
    if energy < e_bins[0]:
        return 0
    elif energy > e_bins[-1]:
        return len(e_bins)-1
    else:
        return np.where(energy > e_bins)[0][-1]


def scatter_absorb(energy):
    """
    Returns if the photon is absorbed or scattered given an energy
    Input
    - Energy:       Energy of the photon    (keV)

    Output
    - True: scattered
    - False: absorbed
    """
    if energy < 1e-3:
        P = 0.5
    else:
        P = 0.5*(energy/1e3)**-3
    x = random()
    if x < P:
        return True
    else:
        return False


# Oh hi there! Hope you have a nice day
