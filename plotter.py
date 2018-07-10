# -*- coding: utf-8 -*-
#
#   plotter.py
#   Author: Max Briel
#
#   Functions to plot several input spectra and distribution of the
#   spectrum generating code. Also includes plotting the soft and
#   hard state from data files.



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib import rc
from scipy.integrate import simps, quad


from generators import *
from synchrotron import *
from helper_functions import *

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


def plot_SS():
    """
    Plot the soft state from a data file
    """
    cwd = os.getcwd()
    file = cwd+"/report_plot_data/Spectrum"
    # Make soft state plot
    data = pd.read_csv(file+"/Soft state.txt",'\t' )
    data.x = [float(i.replace(",", ".")) for i in data.x.values]
    data.y = [float(i.replace(",", ".")) for i in data.y.values]
    # Make sure the data is sorted correctly before plotting
    x, y = zip(*sorted(zip(data.x.values, data.y.values), key= lambda x: x[0]))
    plt.plot(x, y, linewidth=2, color="blue")
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.title("Cygnus X-1 Soft State")
    plt.xlabel(r"E [keV]")
    plt.ylabel(r"E F$_{E}$ [keV cm$^{-2}$ s$^{-1}$]")
    plt.ylim([0.01,25])
    plt.xlim([0.2,5e4])
    plt.savefig(file+"/soft_state.png", dpi=300, bbox_inches="tight")
    print("Saved soft state plot in %s" % file)
    return None

def plot_HS():
    """
    Plot the Hard State from a data files
    """
    cwd = os.getcwd()
    file = cwd+"/report_plot_data/Spectrum"
    # Make hard state plot
    data = pd.read_csv(file+"/Hard state.txt",'\t' )
    data.x = [float(i.replace(",", ".")) for i in data.x.values]
    data.y = [float(i.replace(",", ".")) for i in data.y.values]
    # Make sure the data is sorted correctly before plotting
    x, y = zip(*sorted(zip(data.x.values, data.y.values), key= lambda x: x[0]))
    plt.plot(x, y, linewidth=2, color="red")
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.title("Cygnus X-1 Hard State")
    plt.xlabel(r"E [keV]")
    plt.ylabel(r"E F$_{E}$ [keV cm$^{-2}$ s$^{-1}$]")
    plt.ylim([0.01,25])
    plt.xlim([0.2,5e4])
    plt.savefig(file+"/hard_state.png", dpi=300, bbox_inches="tight")
    print("Saved hard state plot in %s" % file)
    plt.clf()
    return None


def plot_E_dist():
    """
    Output the energy counts per logmarithmic bin to check if it's flat
    """
    rnd_E = g.rnd_E_generator(1e-15,np.log(1e15/1e-15))
    e_bins = np.logspace(-15,15,100)
    energies = [rnd_E() for i in range(1000000)]
    plt.hist(energies, e_bins)
    plt.title("number of photons per logmarithmic bin")
    plt.xlabel("E [keV]")
    plt.ylabel("counts")
    plt.gca().set_xscale("log")
    plt.savefig("report_plot_data/energy_counts.png", dpi=300)
    print "Saved energy counts in ", os.getcwd()
    plt.clf()
    return None



def nr_photon_bb(counter, e_bins, radius, distance, nr_ph):
    """
    To plot only the BB: Set TE to a value below 512 keV and
    turn off Compton Scattering. Then run create_spectrum to make the
    photon flux spectrum and input it into this funciton to get the photon flux
    at earth.
    """
    plt.hist(x=e_bins[:-1],
             bins=e_bins,
             weights=np.array(counter.values())*(radius/distance)**2)
    plt.plot(np.logspace(-15,15,1000),
            np.array(
            [nr_ph(i) for i in np.logspace(-15,15,1000)])*(radius/distance)**2,
            label = "analytical")

    plt.title(r"Blackbody N$_{ph}$ MC vs Analytical at Earth")
    plt.xlabel(r"E [keV]")
    plt.ylabel(r"N$_{ph}$ [photons cm$^{-2}$ s$^{-1}$ keV$^{-1}$]")
    plt.ylim([1e-1, 10])
    plt.xlim([1e-4, 1e2])
    plt.legend()
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    plt.savefig("report_plot_data/N_ph_bb.png", dpi=300)
    print "Outputted the Blackbody photon flux in report_plot_data/"
    plt.clf()
    return None

def nr_photons_sync(counter, e_bins, radius, distance, TE, P_corona):
    """
    Plot the synchrotron input field.
    Set TE to a value above 512 keV,
    and turn off IC scattering for initial field.
    """
    n_e = electron_density(radius, TE, P_corona)
    B = magnetic_field(TE, n_e)
    vL = 5e-10*B/(2*np.pi*(const.m_e*const.c).to("g cm/s").value)
    pdf_e = MJ_generator(TE)
    nr_electrons = electron_number(n_e, pdf_e)
    emissivity = synchrotron_emissivity_generator(B, nr_electrons, vL)
    nr_photons = nrph_sync_generator(emissivity, radius)

    plt.hist(x=e_bins[:-1],
             bins=e_bins,
             weights=np.array(counter.values())*(radius/distance)**2)
    plt.loglog(e_bins,
               [nr_photons(i)* (radius/distance)**2 for i in e_bins],
               label="analytical")

    plt.title(r"Synchrotron N$_{ph}$ MC vs Analytical at Earth")
    plt.xlabel(r"E [keV]")
    plt.ylabel(r"N$_{ph}$ [photons cm$^{-2}$ s$^{-1}$ keV$^{-1}$]")
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    plt.ylim([1e-2, 1e3])
    plt.xlim([1e-9, 1e0])
    plt.legend()
    plt.savefig("report_plot_data/N_ph_sync.png", dpi=300)
    print "Outputted the synchrotron photon flux in report_plot_data/"
    plt.clf()
    return None


def plot_spectrum(bb, sync, e_bins, radius, distance):
    """
    Plot the full spectrum
    """
    if sync == None:          # allow plotting if no synchrotron present
        sync = Counter({i:0 for i in range(len(e_bins)-1)})

    values = np.array(bb.values())
    flux_bb = values *e_bins[:-1]**2
    values = np.array(sync.values())
    flux_sync = values * e_bins[:-1]**2

    plt.hist(x=e_bins[:-1],
             bins=e_bins,
             weights=flux_bb*(radius/distance)**2,
             label="BB")
    plt.hist(x=e_bins[:-1],
             bins=e_bins,
             weights=flux_sync*(radius/distance)**2,
             label="Synchrotron")

    plt.title(r"$EF_E$ MC spectrum at Earth")
    plt.xlabel(r"E [keV]")
    plt.ylabel(r"E$F_E$ [keV cm$^{-2}$ s$^{-1}$]")
    plt.xlim([1e-9, 1e8])
    plt.ylim(1e-2)
    plt.legend()
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    plt.savefig("report_plot_data/Spectrum.png", dpi=300)
    print "Outputted a Spectrum in report_plot_data/"
    plt.clf()
    return None

def MJ_dist():
    """
    Plot the Maxwell-Juttner distribution for several different
    electron temperatures.
    """
    TE = 512. #keV
    fact = [0.1,1,10,100]
    gammas = np.logspace(0,10,10000)
    for f in fact:
        pdf_e = MJ_generator(TE*f)
        plt.loglog(gammas,
                   [pdf_e(i) for i in gammas],
                   label = r"kT$_e$/mc$^2$ = %.1f" %(TE*f/512))

    plt.title(r"Probability distribution of electrons")
    plt.xlabel(r"Gamma")
    plt.ylabel(r"Probability")
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    plt.legend()
    plt.ylim([1e-10, 1e1])
    plt.xlim([0.9,1e4])
    plt.savefig("report_plot_data/MJ_dist.png", dpi=300)
    plt.clf()
    return None

def plot_spectrum_ss(bb, sync, e_bins, radius, distance):
    """
    Plot the spectrum and include the soft state data in the plot
    """
    if sync == None:          # allow plotting if no synchrotron present
        sync = Counter({i:0 for i in range(len(e_bins)-1)})

    values = np.array(bb.values())
    flux_bb = values *e_bins[:-1]**2
    values = np.array(sync.values())
    flux_sync = values * e_bins[:-1]**2

    plt.hist(x=e_bins[:-1],
             bins=e_bins,
             weights=flux_bb*(radius/distance)**2,
             label="MC BB")
    plt.hist(x=e_bins[:-1],
             bins=e_bins,
             weights=flux_sync*(radius/distance)**2,
             label="MC Synchrotron")

    cwd = os.getcwd()
    file = cwd+"/report_plot_data/Spectrum"
    # Make soft state plot
    data = pd.read_csv(file+"/Soft state.txt",'\t' )
    data.x = [float(i.replace(",", ".")) for i in data.x.values]
    data.y = [float(i.replace(",", ".")) for i in data.y.values]
    # Make sure the data is sorted correctly before plotting
    x, y = zip(*sorted(zip(data.x.values, data.y.values), key= lambda x: x[0]))
    plt.plot(x, y, linewidth=2, color="red", label="data")


    plt.title(r"$EF_E$ MC Spectrum at Earth")
    plt.xlabel(r"E [keV]")
    plt.ylabel(r"E$F_E$ [keV cm$^{-2}$ s$^{-1}$]")
    plt.xlim([1e-3, 1e5])
    plt.ylim(1e-2)
    plt.legend()
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    plt.savefig("report_plot_data/Spectrum_w_SS.png", dpi=300)
    print "Outputted a Spectrum with the data in report_plot_data/"
    plt.clf()
    return None



def plot_spectrum_hs(bb, sync, e_bins, radius, distance):
    """
    Plot the spectrum and include the hard state data
    """
    if sync == None:          # allow plotting if no synchrotron present
        sync = Counter({i:0 for i in range(len(e_bins)-1)})

    values = np.array(bb.values())
    flux_bb = values *e_bins[:-1]**2
    values = np.array(sync.values())
    flux_sync = values * e_bins[:-1]**2

    plt.hist(x=e_bins[:-1],
             bins=e_bins,
             weights=flux_bb*(radius/distance)**2,
             label="MC BB")
    plt.hist(x=e_bins[:-1],
             bins=e_bins,
             weights=flux_sync*(radius/distance)**2,
             label="MC Synchrotron")

    cwd = os.getcwd()
    file = cwd+"/report_plot_data/Spectrum"
    # Make soft state plot
    data = pd.read_csv(file+"/Hard state.txt",'\t' )
    data.x = [float(i.replace(",", ".")) for i in data.x.values]
    data.y = [float(i.replace(",", ".")) for i in data.y.values]
    # Make sure the data is sorted correctly before plotting
    x, y = zip(*sorted(zip(data.x.values, data.y.values), key= lambda x: x[0]))
    plt.plot(x, y, linewidth=2, color="red", label="data")


    plt.title(r"$EF_E$ MC Spectrum at Earth")
    plt.xlabel(r"E [keV]")
    plt.ylabel(r"E$F_E$ [keV cm$^{-2}$ s$^{-1}$]")
    plt.xlim([1e-9, 1e5])
    plt.ylim(1e-2)
    plt.legend()
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    plt.savefig("report_plot_data/Spectrum_w_HS.png", dpi=300)
    print "Outputted a Spectrum with the data in report_plot_data/"
    plt.clf()
    return None
