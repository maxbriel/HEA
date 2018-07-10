# -*- coding: utf-8 -*-
#
#   HEA.py
#   Author: Max Briel
#
#   The main function to generate a spectrum using blackbody and
#   synchrotron emission, which is Inverse Compton scattered
#

# Allow usage of Latex
# -----------------------------
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
rc('text', usetex=True)
# -----------------------------

import numpy as np

# Packages for units
from astropy import constants as const
from astropy import units as u

# Helper packages
from collections import Counter

# generator functions
from generators import *

# General helper functions
from helper_functions import *

# Blackbody functions
from blackbody import *

# Synchrotron functions
from synchrotron import *

# Plot functions
import plotter as pl

# -----------------------------------------------------------------------------
#                      MAIN SPECTRUM GENERATION FUNCTION
# -----------------------------------------------------------------------------

def create_spectrum(mass,grav_r, distance, P_corona, TBB, TE, e_min, e_max):
    """
    Create a spectrum of a source as abserved at earth

    Input
    - mass:         Mass of the object                  (Solar Mass)
    - distance:     Distance to the source              (cm)
    - P_corona:     Power output of the corona          (keV/s)
    - TBB:          Blackbody temperature               (keV)
    - TE:           Electron temperature                (keV)
    - e_min:        min sample space boundary energy    (keV)
    - e_max:        max sample space boundary energy    (keV)

    Output:
    - BB_bins:      BB photon flux per bin     (Counter [#photons/cm2/s/keV])
    - sync_bins:    Synchrotron flux per bin   (Counter [#photons/cm2/s/keV])
    - e_bins:       the starting values of each energy bin  (array)
    """

    # Numerical parameters
    nphotons = 100000
    nbins = 1000
    ratio = np.log(e_max/e_min)
    e_bins = np.logspace(np.log10(e_min), np.log10(e_max), nbins)

    # Assume the source has (factor * r_g = radius)
    radius = grav_r*grav_radius(mass)                    # Radius
    n_e = electron_density(radius, TE, P_corona)      # Electron density
    tau = radius*n_e*const.sigma_T.to(u.cm**2).value  # Optical Depth

    # Output data
    print("Radius:\t\t\t %.3g \t (cm)" % radius)
    print("Electron dens:\t\t %.3g \t (1/cm3)" % n_e)
    print("Tau: \t\t\t %.3g" % tau)


    # Setup for the weight method
    rnd_phE = rnd_E_generator(e_min, ratio)     # Random energy generator (1/E)


    # Setup for electron distribution
    pdf_e = MJ_generator(TE)                    # pdf_e(gamma_electron)
    gammamax, ymax = pdf_min_max(pdf_e, e_bins)
    # e_bins is not the electron energy. Its used here, because gammamax
    # will be in this range
    rnd_electron = rnd_electron_generator(pdf_e, gammamax, ymax)


    BB_bins = None
    sync_bins = None

    # Do blackbody compton scattering
    BB_bins = do_BB_comp(nphotons, e_bins, TBB, tau, ratio,
                                                        rnd_phE, rnd_electron)

    # Only do synchrotron if the electrons are relativistic
    if TE > 512.:
        B = magnetic_field(TE, n_e)
        # Larmor frequency
        vL = 5e-10*B/(2*np.pi*(const.m_e*const.c).to("g cm/s").value)
        nr_electrons = electron_number(n_e, pdf_e)
        emissivity = synchrotron_emissivity_generator(B, nr_electrons, vL)

        print("B Field:\t\t %.3g \t (G)" % B)
        print("Larmor frequency: \t %.3g \t (Hz)" % vL)
        vLkeV = vL*const.h.to("keV/Hz").value
        print("Larmor frequency: \t %.3g \t (keV)" % vLkeV)

        sync_bins = do_synchrotron(nphotons, e_bins, radius,
                                tau, ratio, emissivity, rnd_phE, rnd_electron)

    return (BB_bins, sync_bins, e_bins)


# -----------------------------------------------------------------------------
#                  MAIN PARAMETER SETUP AND SIMULATION RUN
# -----------------------------------------------------------------------------


# Physical Parameters
mass = 14.8                 # Mass of object   (Sol Mass)
distance = 1900             # distance         (pc)
Pc_X1 = 3e34                # Corona Power     (erg/s)
TBB = 0.4                   # BB Temp          (keV)
TE = 100                    # Electron Temp    (keV)
grav_r = 5                  # Amount of times the gravitational radius
radius = grav_r * grav_radius(mass)

# Transfer input parameters to other units
distance = (distance*u.pc).to(u.cm).value
Pc = (Pc_X1*u.erg).to(u.keV).value

print("Mass:\t\t\t %.3g \t\t (solar mass)" % mass)
print("Distance:\t\t %.3g \t (cm)" % distance)
print("Corona power input:\t %.3g \t\t (erg/s)" % Pc_X1)
print("Corona power input:\t %.3g \t (keV/s)" % Pc)
print("BB temp:\t\t %.3g \t\t (keV)" % TBB)
print("Electron temp: \t\t %g \t\t (keV)" %TE)

# Monte Carlo Energy boundaries parameters
e_min = 1e-15
e_max = 1e15

# Create the spectrum
BB_bins, sync_bins, bins = create_spectrum(mass, grav_r, distance,
                                           Pc, TBB, TE, e_min, e_max)

# -----------------------------------------------------------------------------
#                        FUNCTION CALS TO PLOT FIGURES
# -----------------------------------------------------------------------------
#   Uncomment to plot. Some internal functions might have to be altered.
#   Please check the function for the exact changes.

# Plots the initial energy distribution
#pl.plot_E_dist()

# Plot the states from data files
#pl.plot_HS()
#pl.plot_SS()

# Plot the electron distibution
#pl.MJ_dist()

# Plot the blackbody and/or synchrotron spectrum
#pl.nr_photon_bb(BB_bins, bins, radius, distance, BB_generator(TBB))
#pl.nr_photons_sync(sync_bins, bins, radius, distance, TE, Pc)

# Plotting the full spectrum
pl.plot_spectrum(BB_bins, sync_bins, bins, radius, distance)


# Plot the spectrum compared to the soft and hard state
#pl.plot_spectrum_ss(BB_bins, sync_bins, bins, radius, distance)
#pl.plot_spectrum_hs(BB_bins, sync_bins, bins, radius, distance)
