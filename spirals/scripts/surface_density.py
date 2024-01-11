import os
import re
import time
import warnings
import matplotlib
import plot_tools

matplotlib.use('Agg')

import numpy as np
import cmasher as cmr
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

    try:
        group_numbers = np.load(data_path + 'group_numbers.npy')
    except FileNotFoundError:
        return None
    if len(group_numbers) == 0:
        return None

    subgroup_numbers = np.load(data_path + 'subgroup_numbers.npy')
    glx_disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')

    # Normalise the disc fractions #
    chi = 0.5 * (1 - np.cos(np.pi / 6))
    glx_disc_fractions_IT20 = np.divide(1, 1 - chi) * (glx_disc_fractions_IT20 - chi)
    mask_disc_fraction, = np.where(glx_disc_fractions_IT20 > 0)

    for group_number, subgroup_number in zip(group_numbers[mask_disc_fraction], subgroup_numbers[mask_disc_fraction]):
        stellar_data_tmp = np.load(data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' +
                                   str(subgroup_number) + '.npy', allow_pickle=True)
        gaseous_data_tmp = np.load(data_path + 'gaseous_data_tmps/gaseous_data_tmp_' + str(group_number) + '_' +
                                   str(subgroup_number) + '.npy', allow_pickle=True)
        blackhole_data_tmp = np.load(data_path + 'blackhole_data_tmps/blackhole_data_tmp_' + str(group_number) + '_' +
                                     str(subgroup_number) + '.npy', allow_pickle=True)
        stellar_data_tmp, gaseous_data_tmp, blackhole_data_tmp = stellar_data_tmp.item(), gaseous_data_tmp.item(), \
            blackhole_data_tmp.item()

        # Rotate coordinates and velocities of stellar particles so the galactic angular momentum points along the x axis #
        coordinates, velocities, prc_angular_momentum, glx_angular_momentum = RotateCoordinates.rotate_Jz(
            stellar_data_tmp)

        # Generate the figure and define its parameters #
        figure = plt.figure(figsize=(30, 20))
        gs = gridspec.GridSpec(nrows=2, ncols=3, wspace=0.25)
        axis00, axis01, axis02 = figure.add_subplot(gs[0, 0]), figure.add_subplot(gs[0, 1]), figure.add_subplot(
            gs[0, 2])
        axis10, axis11, axis12 = figure.add_subplot(gs[1, 0]), figure.add_subplot(gs[1, 1]), figure.add_subplot(
            gs[1, 2])

        plot_tools.set_axes(axis00, xlim=[-1, 1], ylim=[-1, 1], xlabel=r'$x/\mathrm{kpc}$', ylabel=r'$y/\mathrm{kpc}$',
                            aspect=None, which='major', size=30)
        plot_tools.set_axes(axis01, xlim=[-1, 1], ylim=[-1, 1], xlabel=r'$x/\mathrm{kpc}$', ylabel=r'$z/\mathrm{kpc}$',
                            aspect=None, which='major', size=30)
        plot_tools.set_axes(axis02, xlim=[-1, 1], ylim=[-1, 1], xlabel=r'$y/\mathrm{kpc}$', ylabel=r'$z/\mathrm{kpc}$',
                            aspect=None, which='major', size=30)

        plot_tools.set_axes(axis10, xlim=[-1, 1], ylim=[-1, 1], xlabel=r'$x/\mathrm{kpc}$', ylabel=r'$y/\mathrm{kpc}$',
                            aspect=None, which='major', size=30)
        plot_tools.set_axes(axis11, xlim=[-1, 1], ylim=[-1, 1], xlabel=r'$x/\mathrm{kpc}$', ylabel=r'$z/\mathrm{kpc}$',
                            aspect=None, which='major', size=30)
        plot_tools.set_axes(axis12, xlim=[-1, 1], ylim=[-1, 1], xlabel=r'$y/\mathrm{kpc}$', ylabel=r'$z/\mathrm{kpc}$',
                            aspect=None, which='major', size=30)

        # Add text. #
        figure.text(0.1, 0.9, '$\mathrm{Face-on\; stars}$', fontsize=30, transform=axis00.transAxes)
        figure.text(0.1, 0.9, '$\mathrm{Edge-on\; stars}$', fontsize=30, transform=axis01.transAxes)
        figure.text(0.1, 0.9, '$\mathrm{Side-on\; stars}$', fontsize=30, transform=axis02.transAxes)
        figure.text(0.1, 0.9, '$\mathrm{Face-on\; gas}$', color='white', fontsize=30, transform=axis10.transAxes)
        figure.text(0.1, 0.9, '$\mathrm{Edge-on\; gas}$', color='white', fontsize=30, transform=axis11.transAxes)
        figure.text(0.1, 0.9, '$\mathrm{Side-on\; gas}$', color='white', fontsize=30, transform=axis12.transAxes)

        # Add text. #
        figure.text(0.1, 0.05, r'$z = %.0f$' '\n' r'$D/T = %.2f$' '\n'
                               r'$\mathrm{log}_{10}(M_\bigstar/ \mathrm{M}_\odot) = %.2f}$'
                    % (int(re.split('_', data_path)[-1][3:4]), stellar_data_tmp['disc_fraction_IT20'],
                       np.log10(np.sum(stellar_data_tmp['Mass']))), fontsize=30, transform=axis00.transAxes)

        figure.text(0.1, 0.05, r'$\mathrm{log}_{10}(M_\bullet / \mathrm{M}_\odot) = %.2f}$' '\n'
                               r'$\mathrm{log}_{10}(M_\mathrm{gas}/ \mathrm{M}_\odot) = %.2f}$'
                    % (np.log10(np.sum(blackhole_data_tmp['BH_Mass'])), np.log10(np.sum(gaseous_data_tmp['Mass']))),
                    color='white', fontsize=30,
                    transform=axis10.transAxes)

        # Plot a circle with each redshift's softening length. #
        # sl = plt.Circle((0, 0), softening, color='w', fill=False)
        # axis00.add_patch(sl)

        cmap = matplotlib.cm.get_cmap('Greys')
        axis00.set_facecolor(cmap(0)), axis01.set_facecolor(cmap(0)), axis02.set_facecolor(cmap(0))
        count, xedges, yedges = np.histogram2d(coordinates[:, 0], coordinates[:, 1], weights=stellar_data_tmp['Mass'],
                                               bins=100, range=[[-1, 1], [-1, 1]])
        axis00.imshow(count.T, norm=matplotlib.colors.LogNorm(), extent=[-1, 1, -1, 1], origin='lower', cmap=cmap,
                      rasterized=True, aspect='equal')

        count, xedges, yedges = np.histogram2d(coordinates[:, 0], coordinates[:, 2], weights=stellar_data_tmp['Mass'],
                                               bins=100, range=[[-1, 1], [-1, 1]])
        axis01.imshow(count.T, norm=matplotlib.colors.LogNorm(), extent=[-1, 1, -1, 1], origin='lower', cmap=cmap,
                      rasterized=True, aspect='equal')

        count, xedges, yedges = np.histogram2d(coordinates[:, 1], coordinates[:, 2], weights=stellar_data_tmp['Mass'],
                                               bins=100, range=[[-1, 1], [-1, 1]])
        axis02.imshow(count.T, norm=matplotlib.colors.LogNorm(), extent=[-1, 1, -1, 1], origin='lower', cmap=cmap,
                      rasterized=True, aspect='equal')

        # Rotate coordinates and velocities of gas particles so the galactic angular momentum points along the x axis #
        # Calculate the angular momentum of the galaxy #
        prc_angular_momentum = stellar_data_tmp['Mass'][:, np.newaxis] * np.cross(stellar_data_tmp['Coordinates'],
                                                                                  stellar_data_tmp[
                                                                                      'Velocity'])  # In Msun kpc km s^-1.
        glx_angular_momentum = np.sum(prc_angular_momentum, axis=0)  # In Msun kpc km s^-1.

        # Define the rotation matrices #
        a = np.matrix([glx_angular_momentum[0], glx_angular_momentum[1], glx_angular_momentum[2]]) / np.linalg.norm(
            [glx_angular_momentum[0], glx_angular_momentum[1], glx_angular_momentum[2]])
        b = np.matrix([0, 0, 1])
        v = np.cross(a, b)
        s = np.linalg.norm(v)
        c = np.dot(a, b.T)
        vx = np.matrix([[0, -v[0, 2], v[0, 1]], [v[0, 2], 0, -v[0, 0]], [-v[0, 1], v[0, 0], 0]])
        transform = np.eye(3, 3) + vx + (vx * vx) * ((1 - c[0, 0]) / s ** 2)

        # Rotate the coordinates and velocities #
        coordinates = np.array([np.matmul(transform, gaseous_data_tmp['Coordinates'][i].T) for i in
                                range(0, len(gaseous_data_tmp['Coordinates']))])[:, 0]

        cmap = matplotlib.cm.get_cmap('inferno')
        axis10.set_facecolor(cmap(0)), axis11.set_facecolor(cmap(0)), axis12.set_facecolor(cmap(0))
        count, xedges, yedges = np.histogram2d(coordinates[:, 0], coordinates[:, 1], weights=gaseous_data_tmp['Mass'],
                                               bins=100, range=[[-1, 1], [-1, 1]])
        axis10.imshow(count.T, norm=matplotlib.colors.LogNorm(), extent=[-1, 1, -1, 1], origin='lower', cmap=cmap,
                      rasterized=True, aspect='equal')

        count, xedges, yedges = np.histogram2d(coordinates[:, 0], coordinates[:, 2], weights=gaseous_data_tmp['Mass'],
                                               bins=100, range=[[-1, 1], [-1, 1]])
        axis11.imshow(count.T, norm=matplotlib.colors.LogNorm(), extent=[-1, 1, -1, 1], origin='lower', cmap=cmap,
                      rasterized=True, aspect='equal')

        count, xedges, yedges = np.histogram2d(coordinates[:, 1], coordinates[:, 2], weights=gaseous_data_tmp['Mass'],
                                               bins=100, range=[[-1, 1], [-1, 1]])
        axis12.imshow(count.T, norm=matplotlib.colors.LogNorm(), extent=[-1, 1, -1, 1], origin='lower', cmap=cmap,
                      rasterized=True, aspect='equal')

        # if not os.path.exists(plots_path): os.makedirs(plots_path)
        # TODO user defined
        plt.savefig('/cosma7/data/dp004/dc-irod1/FLARES/spirals/plots/surface_density_times_two-' + str(tag) + '_' +
                    str(region) + '_' + str(group_number) + '_' + str(subgroup_number) + '.png', bbox_inches='tight')
        plt.close()
    return None


if __name__ == '__main__':

    # Analyse FLARES data #
    # TODO user defined
    redshifts = [8]  # 8, 7, 6, 5]
    tags = ['007_z008p000']  # ['007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000']
    for tag, redshift in zip(tags, redshifts):
        if redshift <= 2.8:
            softening = 0.000474390 / 0.6777 * 1e3  # In pkpc.
        else:
            softening = 0.001802390 / (0.6777 * (1 + redshift)) * 1e3  # In pkpc.

        # for region in ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15',
        #                '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32',
                       # '33', '34', '35', '36', '37', '38', '39']:
        for region in ['00']:
            print('z:', redshift, 'region:', region)

            # Path to load data and to save plots. # TODO user defined
            data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/reruns/times_two/' + region + '/' + tag + '/'
            # data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/JWST/' + region + '/' + tag + '/'
            # plots_path = '/cosma7/data/dp004/dc-irod1/FLARES/spirals/plots/' + region + '/' + tag + '/'
            plot()
