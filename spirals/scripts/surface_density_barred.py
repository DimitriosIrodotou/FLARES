import re
import time
import warnings
import matplotlib
import plot_tools

matplotlib.use('Agg')

import numpy as np
import matplotlib.cbook
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.style as style
from plot_tools import RotateCoordinates

style.use("classic")
obsHubble = 0.70  # [dimensionless]
plt.rcParams.update({'font.family': 'serif'})
date = time.strftime('%d_%m_%y_%H%M')  # Date.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


def plot():
    """
    Plot the surface density of a given galaxy.
    :return: None
    """
    # Load the data. #
    stellar_data_tmp = np.load(data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' +
                               str(subgroup_number) + '.npy', allow_pickle=True)
    stellar_data_tmp = stellar_data_tmp.item()

    # Rotate coordinates and velocities of stellar particles so the galactic angular momentum points along the x axis #
    coordinates, velocities, prc_angular_momentum, glx_angular_momentum = RotateCoordinates.rotate_Jz(
        stellar_data_tmp)

    # Add text. #
    figure.text(0.05, 0.8, r'$z = %.0f \;$'  '\n'  r'$\mathrm{log}_{10}(M_\bigstar/ \mathrm{M}_\odot) = %.2f} \;$'
                % (int(re.split('_', data_path)[-1][3:4]), np.log10(np.sum(stellar_data_tmp['Mass']))), color='white',
                fontsize=25, transform=axis.transAxes)

    cmap = matplotlib.cm.get_cmap('inferno')
    axis.set_facecolor(cmap(0)), axis2.set_facecolor(cmap(0))
    count, xedges, yedges = np.histogram2d(coordinates[:, 0], coordinates[:, 1], weights=stellar_data_tmp['Mass'],
                                           bins=150, range=[[-2, 2], [-2, 2]])
    axis.imshow(count.T, norm=matplotlib.colors.LogNorm(), extent=[-2, 2, -2, 2], origin='lower', cmap=cmap,
                rasterized=True, aspect='equal')

    count, xedges, yedges = np.histogram2d(coordinates[:, 0], coordinates[:, 2], weights=stellar_data_tmp['Mass'],
                                           bins=150, range=[[-2, 2], [-2, 2]])
    axis2.imshow(count.T, norm=matplotlib.colors.LogNorm(), extent=[-2, 2, -2, 2], origin='lower', cmap=cmap,
                 rasterized=True, aspect='equal')

    return None


if __name__ == '__main__':
    # Analyse FLARES data #
    barred_candidates = ['007_z008p000_11_2_0', '008_z007p000_04_8_0', '008_z007p000_11_6_0', '008_z007p000_12_14_0',
                         '009_z006p000_00_2_1', '009_z006p000_01_72_0', '009_z006p000_12_14_0', '009_z006p000_15_3_0',
                         '009_z006p000_21_9_0', '010_z005p000_02_74_0', '010_z005p000_03_24_0', '010_z005p000_08_87_0',
                         '010_z005p000_09_38_0', '010_z005p000_18_24_0', '010_z005p000_21_10_0', '010_z005p000_32_6_0']

    spiral_candidates = ['007_z008p000_00_2_0', '008_z007p000_04_8_0', '009_z006p000_00_2_1', '009_z006p000_00_29_0',
                         '009_z006p000_02_42_0', '009_z006p000_02_4_0', '009_z006p000_01_72_0', '009_z006p000_03_4_0',
                         '009_z006p000_07_15_0', '009_z006p000_10_19_0', '009_z006p000_12_14_0', '009_z006p000_32_6_0',
                         '009_z006p000_21_9_0', '010_z005p000_00_4_1', '010_z005p000_01_13_0', '010_z005p000_02_8_0',
                         '010_z005p000_02_62_0', '010_z005p000_02_74_0', '010_z005p000_04_2_1', '010_z005p000_04_45_0',
                         '010_z005p000_08_4_3', '010_z005p000_09_38_0', '010_z005p000_12_1_3', '010_z005p000_12_61_0',
                         '010_z005p000_15_1_0', '010_z005p000_15_107_0', '010_z005p000_17_25_0', '010_z005p000_18_24_0',
                         '010_z005p000_21_10_0', '010_z005p000_32_6_0']

    # Generate the figure and define its parameters #
    number_of_galaxies = len(barred_candidates)
    figure = plt.figure(figsize=(20, 29))
    gs = gridspec.GridSpec(nrows=8, ncols=4, wspace=0.0, hspace=0.0, height_ratios=[1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5])

    tags, regions, group_numbers, subgroup_numbers = [], [], [], []
    for candidate in barred_candidates:
        tags.append(re.split('_', candidate)[0] + '_' + re.split('_', candidate)[1])
        regions.append(re.split('_', candidate)[2])
        group_numbers.append(re.split('_', candidate)[3]), subgroup_numbers.append(re.split('_', candidate)[4])
    for _, (tag, region, group_number, subgroup_number) in enumerate(
            zip(tags, regions, group_numbers, subgroup_numbers)):
        if _ < int(number_of_galaxies / 4):
            axis = figure.add_subplot(gs[0, _ - 4])
            axis2 = figure.add_subplot(gs[1, _ - 4])
            if _ > 0:
                axis.set_yticks([]), axis2.set_yticks([]), axis.set_yticklabels([]), axis2.set_yticklabels([])
            else:
                axis.set_yticks([-0.5, 0, 0.5]), axis2.set_yticks([-0.25, 0, 0.25])
            axis.set_xticks([]), axis2.set_xticks([]), axis.set_xticklabels([]), axis2.set_xticklabels([])
        elif int(number_of_galaxies / 4) <= _ < int(number_of_galaxies / 2):
            axis = figure.add_subplot(gs[2, _ - 8])
            axis2 = figure.add_subplot(gs[3, _ - 8])
            if _ > 4:
                axis.set_yticks([]), axis2.set_yticks([]), axis.set_yticklabels([]), axis2.set_yticklabels([])
            else:
                axis.set_yticks([-0.5, 0, 0.5]), axis2.set_yticks([-0.25, 0, 0.25])
            axis.set_xticks([]), axis2.set_xticks([]), axis.set_xticklabels([]), axis2.set_xticklabels([])
        elif int(number_of_galaxies / 2) <= _ < int(3 * number_of_galaxies / 4):
            axis = figure.add_subplot(gs[4, _ - 12])
            axis2 = figure.add_subplot(gs[5, _ - 12])
            if _ > 8:
                axis.set_yticks([]), axis2.set_yticks([]), axis.set_yticklabels([]), axis2.set_yticklabels([])
            else:
                axis.set_yticks([-0.5, 0, 0.5]), axis2.set_yticks([-0.25, 0, 0.25])
            axis.set_xticks([]), axis2.set_xticks([]), axis.set_xticklabels([]), axis2.set_xticklabels([])
        elif int(3 * number_of_galaxies / 4) <= _ < int(number_of_galaxies):
            axis = figure.add_subplot(gs[6, _ - 16])
            axis2 = figure.add_subplot(gs[7, _ - 16])
            if _ > 12:
                axis.set_yticks([]), axis2.set_yticks([]), axis.set_yticklabels([]), axis2.set_yticklabels([])
            else:
                axis.set_yticks([-0.5, 0, 0.5]), axis2.set_yticks([-0.25, 0, 0.25])
            axis.set_xticks([]), axis.set_xticklabels([]), axis2.set_xticks([-0.5, 0, 0.5])

        plot_tools.set_axes(axis, xlim=[-1, 1], ylim=[-1, 1], xlabel=r'$x/\mathrm{kpc}$', ylabel=r'$y/\mathrm{kpc}$',
                            aspect=None, which='major', size=30)
        plot_tools.set_axes(axis2, xlim=[-1, 1], ylim=[-0.5, 0.5], xlabel=r'$x/\mathrm{kpc}$',
                            ylabel=r'$z/\mathrm{kpc}$', aspect=None, which='major', size=30)
        # Path to load data and to save plots. #
        data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/JWST/' + region + '/' + tag + '/'
        plot()

    # Save and close  the figure. #
    plt.savefig('/cosma7/data/dp004/dc-irod1/FLARES/spirals/plots/surface_density_barred.png', bbox_inches='tight')
    plt.close()
