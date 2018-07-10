# -*- coding: utf-8 -*-
#
#   generators.py
#   Author: Max Briel
#
#   Several functions that generate random values or
#   setup probability density functions.

from random import random
import numpy as np
from scipy.special import kn   # Bessel function

# Packages for units
from astropy import constants as const
from astropy import units as u

# ---------------------------------------------------------------------------
#                   RANDOM VALUE GENERATORS
# ---------------------------------------------------------------------------

def rnd_E_generator(Emin, ratio):
    """
    Creates a function for a random energy following a 1/x distribution

    Input
    - Emin: minimum energy
    - ratio: log ratio between x_max and x_min

    Output
    - Function: return a random number following a 1/x distribution
    """
    def rnd_E():
        return Emin*np.exp(random()*ratio)
    return rnd_E


def rnd_electron_generator(pdf_e, xmax, ymax):
    """
    Return a function that gives a random electron back

    Input
    - pdf_e:        Probability density function of electrons
    - xmax:         maximum gamma value
    - ymax:         maximum probability

    Output
    - function:     Random electron energy
    """
    def rnd_electron():
        """
        Returns a random electron energ following a given pdf
        """
        x = random()*xmax   # x
        y = random()*ymax   # y has to be between 0 and 1
        while y > pdf_e(x) and x != 0:
            x = random()*xmax
            y = random()*ymax
        return x
    return rnd_electron


# ---------------------------------------------------------------------------
#               PROBABILITY DENSITY FUNCTION GENERATORS
# ---------------------------------------------------------------------------

def MJ_generator(T_e):
    """
    Generate a function that gives the probability density function
    that follows a Maxwell-Juptner distribution at a given temperature

    Input
    - T_e: Electron temperature

    Output:
    - Function: pdf at given temperature
    """
    T_e = T_e * u.keV
    theta = (T_e/(const.m_e * const.c**2)).to("").value
    bessel = kn(2,1./theta)

    def pdf_e(gamma):
        # if gamma smaller than 1, then non-relativistic. Set pdf_e to zero
        if gamma**2 <= 1:
            return 0
        else:
            beta = np.sqrt(1.-1./(gamma**2))
            return (gamma**2 *beta)/(theta*bessel) * np.exp(-(gamma/theta))
    return pdf_e


def synchrotron_emissivity_generator(B, electron_density_fn, vL):
    """
    Create a function for the synchrotron emissivity

    Input:
    - B:                Magnetic Field          (G)
    - electron_number:  number of electrons     (#)
    - vL:               Larmor frequency        (Hz)

    Output:
    - Function:         emissivity function
    """
    f1 = ((const.sigma_T.to("cm2").value * const.c.to("cm/s").value * B**2)
        /((48*np.pi**2 * vL)))
    f2 = 4./3. * const.h.to(u.keV/u.Hz).value*vL
    def synchrotron_emmisivity(E):
        """
        Returns synchrotron emissivity of a given gamma

        Input
        - E:            Synchrotron energy (keV)

        Output
        - emissivity:   sychrotron emissivity   (erg/(cm3 sr s Hz))
        """
        return f1 * np.sqrt(E/f2) * electron_density_fn(np.sqrt(E/f2))
    return synchrotron_emmisivity
