import os
import time
import warnings
import matplotlib
import plot_tools

matplotlib.use('Agg')

import numpy as np
import healpy as hlp
import matplotlib.cbook
import astropy.units as u
import matplotlib.pyplot as plt

from matplotlib import gridspec
from astropy_healpix import HEALPix
from plot_tools import RotateCoordinates

date = time.strftime('%d_%m_%y_%H%M')  # Date.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class SampleSpatialDistribution:
    """
    For a sample of galaxies create: a spatial distribution of the face-on and edge-on projections plot.
    """


    def __init__(self, tag):
        """
        A constructor method for the class.
        :param tag: redshift directory.
        """
        group_numbers = [11, 37, 44, 158]
        regions = ['00', '00', '00', '00']

        # Generate the figure and define its parameters #
        figure = plt.figure(figsize=(20, 20))

        gs = gridspec.GridSpec(5, 4, wspace=0.3, hspace=0.3, height_ratios=[0.1, 1, 1, 1, 1])
        axiscbar = figure.add_subplot(gs[0, :])
        axis10, axis11, axis12, axis13 = figure.add_subplot(gs[1, 0]), figure.add_subplot(gs[1, 1]), figure.add_subplot(gs[1, 2]), figure.add_subplot(
            gs[1, 3])
        axis20, axis21, axis22, axis23 = figure.add_subplot(gs[2, 0]), figure.add_subplot(gs[2, 1]), figure.add_subplot(gs[2, 2]), figure.add_subplot(
            gs[2, 3])
        axis30, axis31, axis32, axis33 = figure.add_subplot(gs[3, 0]), figure.add_subplot(gs[3, 1]), figure.add_subplot(gs[3, 2]), figure.add_subplot(
            gs[3, 3])
        axis40, axis41, axis42, axis43 = figure.add_subplot(gs[4, 0]), figure.add_subplot(gs[4, 1]), figure.add_subplot(gs[4, 2]), figure.add_subplot(
            gs[4, 3])

        for axis in [axis10, axis12, axis20, axis22, axis30, axis32, axis40, axis42]:
            plot_tools.set_axis(axis, xlabel=r'$\mathrm{x/kpc}$', ylabel=r'$\mathrm{y/kpc}$', aspect=None, size=20)
        for axis in [axis11, axis13, axis21, axis23, axis31, axis33, axis41, axis43]:
            plot_tools.set_axis(axis, xlabel=r'$\mathrm{x/kpc}$', ylabel=r'$\mathrm{z/kpc}$', aspect=None, size=20)

        all_axes = [[axis10, axis11, axis12, axis13], [axis20, axis21, axis22, axis23], [axis30, axis31, axis32, axis33],
                    [axis40, axis41, axis42, axis43]]

        for group_number, region, axes in zip(group_numbers, regions, all_axes):  # Loop over all masked haloes.
            for subgroup_number in range(0, 1):  # Get centrals only.
                start_local_time = time.time()  # Start the local time.

                # Load data from numpy arrays #
                stellar_data_tmp = np.load(
                    data_path + region + '/' + tag + '/stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy',
                    allow_pickle=True)
                stellar_data_tmp = stellar_data_tmp.item()
                print('Loaded data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
                print('–––––––––––––––––––––––––––––––––––––––––––––')

                # Plot the data #
                start_local_time = time.time()  # Start the local time.

                im = self.plot(axes, stellar_data_tmp, group_number)
                print('Plotted data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
                print('–––––––––––––––––––––––––––––––––––––––––––––')

        # Add color bar #
        plot_tools.create_colorbar(axiscbar, im, r'$\mathrm{log_{10}(\Sigma_{\bigstar}/(M_\odot\,kpc^{-2}))}$', 'horizontal', extend='both', size=20)
        # Add text #
        plt.text(0.17, 1.1, r'$\mathrm{Disc\;face-on}$', fontsize=20, transform=axis10.transAxes)
        plt.text(0.17, 1.1, r'$\mathrm{Disc\;edge-on}$', fontsize=20, transform=axis11.transAxes)
        plt.text(0.05, 1.1, r'$\mathrm{Spheroid\;face-on}$', fontsize=20, transform=axis12.transAxes)
        plt.text(0.05, 1.1, r'$\mathrm{Spheroid\;edge-on}$', fontsize=20, transform=axis13.transAxes)

        # Save and close the figure #
        plt.savefig(plots_path + 'SSD' + '-' + date + '.png', bbox_inches='tight')
        plt.close()
        print('Finished MultipleDecomposition for ' + tag + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')


    @staticmethod
    def plot(axes, stellar_data_tmp, group_number):
        """
        Plot the spatial distribution of the face-on and edge-on projections.
        circularity distribution.
        :param axes: set of axes
        :param stellar_data_tmp: from read_add_attributes.py.
        :param group_number: from read_add_attributes.py.
        :return: None
        """

        # Rotate coordinates and velocities of stellar particles so the galactic angular momentum points along the x axis #
        coordinates, velocities, prc_angular_momentum, glx_angular_momentum = RotateCoordinates.rotate_Jz(stellar_data_tmp)

        # Calculate the ra and dec of the (unit vector of) angular momentum for each particle #
        prc_unit_vector = prc_angular_momentum / np.linalg.norm(prc_angular_momentum, axis=1)[:, np.newaxis]
        ra = np.degrees(np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0]))
        dec = np.degrees(np.arcsin(prc_unit_vector[:, 2]))

        # Plot a HEALPix histogram #
        nside = 2 ** 4  # Define the resolution of the grid (number of divisions along the side of a base-resolution grid cell).
        hp = HEALPix(nside=nside)  # Initialise the HEALPix pixelisation class.
        indices = hp.lonlat_to_healpix(ra * u.deg, dec * u.deg)  # Create list of HEALPix indices from particles' ra and dec.
        densities = np.bincount(indices, minlength=hp.npix)  # Count number of data points in each HEALPix grid cell.

        # Perform a top-hat smoothing on the densities #
        smoothed_densities = np.zeros(hp.npix)
        # Loop over all grid cells #
        for i in range(hp.npix):
            mask = hlp.query_disc(nside, hlp.pix2vec(nside, i), np.pi / 6.0)  # Do a 30degree cone search around each grid cell.
            smoothed_densities[i] = np.mean(densities[mask])  # Average the densities of the ones inside and assign this value to the grid cell.

        # Find the location of density maximum and plot its positions and the ra (lon) and dec (lat) of the galactic angular momentum #
        index_densest = np.argmax(smoothed_densities)
        lon_densest = (hp.healpix_to_lonlat([index_densest])[0].value + np.pi) % (2 * np.pi) - np.pi
        lat_densest = (hp.healpix_to_lonlat([index_densest])[1].value + np.pi / 2) % (2 * np.pi) - np.pi / 2

        # Calculate the disc mass fraction as the mass within 30 degrees from the densest grid cell #
        angular_theta_from_densest = np.arccos(
            np.sin(lat_densest) * np.sin(np.arcsin(prc_unit_vector[:, 2])) + np.cos(lat_densest) * np.cos(np.arcsin(prc_unit_vector[:, 2])) * np.cos(
                lon_densest - np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0])))  # In radians.

        # Plot the 2D surface density projection and scatter for the disc #
        disc_mask, = np.where(angular_theta_from_densest < (np.pi / 6.0))
        weights = stellar_data_tmp['Mass'][disc_mask]
        vmin, vmax = 6, 8

        cmap = matplotlib.cm.get_cmap('nipy_spectral_r')
        count, xedges, yedges = np.histogram2d(coordinates[disc_mask, 0], coordinates[disc_mask, 1], weights=weights, bins=100,
                                               range=[[-30, 30], [-30, 30]])
        im = axes[0].imshow(np.log10(count.T), extent=[-30, 30, -30, 30], origin='lower', cmap=cmap,  rasterized=True,
                            aspect='equal')

        count, xedges, yedges = np.histogram2d(coordinates[disc_mask, 0], coordinates[disc_mask, 2], weights=weights, bins=100,
                                               range=[[-30, 30], [-30, 30]])
        axes[1].imshow(np.log10(count.T), extent=[-30, 30, -30, 30], origin='lower', cmap=cmap,  rasterized=True, aspect='equal')

        # Plot the 2D surface density projection and scatter for the bulge #
        bulge_mask, = np.where(angular_theta_from_densest > (np.pi / 6.0))

        weights = stellar_data_tmp['Mass'][bulge_mask]
        count, xedges, yedges = np.histogram2d(coordinates[bulge_mask, 0], coordinates[bulge_mask, 1], weights=weights, bins=100,
                                               range=[[-30, 30], [-30, 30]])
        axes[2].imshow(np.log10(count.T), extent=[-30, 30, -30, 30], origin='lower', cmap=cmap,  rasterized=True, aspect='equal')

        count, xedges, yedges = np.histogram2d(coordinates[bulge_mask, 0], coordinates[bulge_mask, 2], weights=weights, bins=100,
                                               range=[[-30, 30], [-30, 30]])
        axes[3].imshow(np.log10(count.T), extent=[-30, 30, -30, 30], origin='lower', cmap=cmap,  rasterized=True, aspect='equal')

        plt.text(-0.2, 1.1, str(group_number), color='red', fontsize=20, transform=axes[0].transAxes)  # Add text.
        return im


if __name__ == '__main__':
    tag = '011_z004p770'
    data_path = '/cosma7/data/dp004/dc-irod1/FLARES/paper/data/'  # Path to save/load data.
    plots_path = '/cosma7/data/dp004/dc-irod1/FLARES/paper/plots/'  # Path to save plots.
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    x = SampleSpatialDistribution(tag)
