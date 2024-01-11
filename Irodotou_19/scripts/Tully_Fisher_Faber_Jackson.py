import os
import re
import time
import warnings
import plot_tools
import matplotlib.cbook

matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import gridspec

date = time.strftime('%d_%m_%y_%H%M')  # Date.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class TullyFisherFaberJackson:
    """
    For all galaxies create: a Tully-Fisher and Faber-Jackson relation plot.
    """


    def __init__(self, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        # Load the data #
        start_local_time = time.time()  # Start the local time.

        glx_stellar_masses_all = np.load(data_path + 'glx_stellar_masses_all.npy')
        glx_stellar_masses_all = np.array([item for sublist in glx_stellar_masses_all for item in sublist])

        glx_rotationals_all = np.load(data_path + 'glx_rotationals_all.npy')
        glx_rotationals_all = np.array([item for sublist in glx_rotationals_all for item in sublist])

        glx_disc_fractions_IT20_all = np.load(data_path + 'glx_disc_fractions_IT20_all.npy')
        glx_disc_fractions_IT20_all = np.array([item for sublist in glx_disc_fractions_IT20_all for item in sublist])

        # Normalise the disc fractions #
        chi = 0.5 * (1 - np.cos(np.pi / 6))
        glx_disc_fractions_IT20_all = np.divide(1, 1 - chi) * (glx_disc_fractions_IT20_all - chi)
        print('Loaded data  for ' + tag + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')

        # Plot the data #
        start_local_time = time.time()  # Start the local time.

        self.plot(glx_stellar_masses_all, glx_rotationals_all, glx_disc_fractions_IT20_all)
        print('Plotted data for ' + tag + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')

        print('Finished TullyFisherFaberJackson for ' + tag + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')


    @staticmethod
    def plot(glx_stellar_masses_all, glx_rotationals_all, glx_disc_fractions_IT20_all):
        """
        Plot the Tully-Fisher and Faber-Jackson relation.
        :param glx_stellar_masses_all: defined as the mass of all stellar particles within 30kpc from the most bound particle.
        :param glx_rotationals_all: defined as the rotational velocity for the whole galaxy.
        :param glx_disc_fractions_IT20_all: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest
        grid cell.
        :return: None
        """
        # Generate the figure and define its parameters #
        figure = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(3, 2, wspace=0.05, hspace=0.05, height_ratios=[0.05, 1, 1])
        axiscbar = figure.add_subplot(gs[0, :])
        axis10, axis11 = figure.add_subplot(gs[1, 0]), figure.add_subplot(gs[1, 1])
        axis20, axis21 = figure.add_subplot(gs[2, 0]), figure.add_subplot(gs[2, 1])

        plot_tools.set_axis(axis10, ylabel=r'$\mathrm{log_{10}(M_{\bigstar}/M_{\odot})}$', aspect=None)
        plot_tools.set_axis(axis11,  aspect=None)
        plot_tools.set_axis(axis20,  xlabel=r'$\mathrm{log_{10}(V_{rot}/(km\;s^{-1}))}$',
                            ylabel=r'$\mathrm{log_{10}(M_{\bigstar}/M_{\odot})}$', aspect=None)
        plot_tools.set_axis(axis21, xlabel=r'$\mathrm{log_{10}(\sigma_{0,e}/(km\;s^{-1}))}$', aspect=None)
        # for axis in [axis11, axis21]:
        #     axis.set_yticklabels([])
        # for axis in [axis10, axis11]:
        #     axis.set_xticklabels([])

        # Plot the Tully-Fisher relations #
        sc = axis20.scatter(np.log10(glx_rotationals_all), np.log10(glx_stellar_masses_all), c=glx_disc_fractions_IT20_all, s=10, cmap='seismic_r',
                            vmin=0, vmax=1)

        # Read and plot observational data from AZF08, TEA11 and OCB20 #
        # AZF08 = np.genfromtxt('./observational_data/AZF_0807.0636/Figure1.csv', delimiter=',', names=['Vrot', 'Mstar'])
        # OCB20_TF_DD = np.genfromtxt('./observational_data/OCB_2005.06474/Figure8_TF_DD.csv', delimiter=',', names=['Vrot', 'Mstar'])
        # OCB20_TF_discs = np.genfromtxt('./observational_data/OCB_2005.06474/Figure8_TF_discs.csv', delimiter=',', names=['Vrot', 'Mstar'])
        # OCB20_TF_bulges = np.genfromtxt('./observational_data/OCB_2005.06474/Figure8_TF_bulges.csv', delimiter=',', names=['Vrot', 'Mstar'])
        # OCB20_FJ_BD = np.genfromtxt('./observational_data/OCB_2005.06474/Figure8_FJ_BD.csv', delimiter=',', names=['sigma', 'Mstar'])
        # OCB20_FJ_discs = np.genfromtxt('./observational_data/OCB_2005.06474/Figure8_FJ_discs.csv', delimiter=',', names=['sigma', 'Mstar'])
        # OCB20_FJ_bulges = np.genfromtxt('./observational_data/OCB_2005.06474/Figure8_FJ_bulges.csv', delimiter=',', names=['sigma', 'Mstar'])
        #
        # axis10.scatter(AZF08['Mstar'], AZF08['Vrot'], color='cyan', marker='s', s=15, label=r'$\mathrm{Avila-Reese+08}$')
        # axis10.plot(OCB20_TF_DD['Vrot'], OCB20_TF_DD['Mstar'], color='cyan', label=r'$\mathrm{Oh+20:B/T<0.2}$')
        # axis11.plot(OCB20_FJ_BD['sigma'], OCB20_FJ_BD['Mstar'], color='orange', label=r'$\mathrm{Oh+20:B/T>0.8}$')
        # axis20.plot(OCB20_TF_discs['Vrot'], OCB20_TF_discs['Mstar'], color='cyan', label=r'$\mathrm{Oh+20:discs}$')
        # axis20.plot(OCB20_TF_bulges['Vrot'], OCB20_TF_bulges['Mstar'], color='orange', label=r'$\mathrm{Oh+20:bulges}$')
        # axis21.plot(OCB20_FJ_discs['sigma'], OCB20_FJ_discs['Mstar'], color='cyan', label=r'$\mathrm{Oh+20:discs}$')
        # axis21.plot(OCB20_FJ_bulges['sigma'], OCB20_FJ_bulges['Mstar'], color='orange', label=r'$\mathrm{Oh+20:bulges}$')

        # Create the legend and save the figure #
        axis10.legend(loc='upper left', fontsize=12, frameon=False, numpoints=1)
        for axis in [axis11, axis20, axis21]:
            axis.legend(loc='upper left', fontsize=12, frameon=False, numpoints=1, ncol=2)
        plt.savefig(plots_path + 'TFFJ' + '-' + date + '.png', bbox_inches='tight')
        plt.close()
        return None


if __name__ == '__main__':
    tag = '011_z004p770'
    data_path = '/cosma7/data/dp004/dc-irod1/FLARES/paper/data/' + tag + '/'  # Path to save/load data.
    plots_path = '/cosma7/data/dp004/dc-irod1/FLARES/paper/plots/'  # Path to save plots.
    x = TullyFisherFaberJackson(tag)
