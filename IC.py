# -*- coding: utf-8 -*-
#
#   IC.py
#   Author: Max Briel
#
#   Code to Inverse Compton scatter an electron using a Thompson Approximation
#

import numpy as np
from random import random


def upscatter(e_photon, theta, gamma):
    """
    calculates the upscattered photon's energy.

    Input
    - e_photon:     initial photon energy  (keV)
    - theta:        angle photon-electron direction
    - gamma:        gamma factor of situation

    Ouput:
    - Upscattered electron energy:      (keV)
    """
    beta = np.sqrt(1-1./gamma**2)
    # Generate incoming angle
    cos_theta = np.cos(theta)

    # Boost to electron rest frame
    e_photon_p = e_photon*gamma*(1.-beta*cos_theta)
    cos_theta_p = (cos_theta-beta)/(1.-beta*cos_theta)

    # Thomson scattering
    e_photon_out_p = e_photon_p

    # Generate a random scattering angle with a distribution proportional
    # to the cross-section. This follows a 1 +cos^2 distribution
    a = 2.0*random()-1.
    b = 2.0*random()
    if b >= 1.0+pow(a,2):
        a = -1.0+2.0*random()
        b = 2.0*random()
    cos_a_p = a

    # Random angle for the azimuthal angle in the rest frame
    cos_phi_p = np.cos(random()*2.0*np.pi)

    sin_theta_p = np.sqrt(1.0-pow(cos_theta_p,2))		# outgoing theta prime
    sin_alpha_p = np.sqrt(1.0-pow(cos_a_p,2))		    # outgoing alpha prime

    # Project outgoing photon such that it's towards the electron's movement
    cos_theta1_p = cos_theta_p*cos_a_p - sin_alpha_p*sin_theta_p*cos_phi_p

    # Transform back to the lab fame
    return e_photon_out_p*gamma*(1+beta*cos_theta1_p)
