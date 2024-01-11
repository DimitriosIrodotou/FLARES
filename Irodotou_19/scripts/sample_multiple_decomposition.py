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


class SampleMultipleDecomposition:
    """
    For a sample of galaxies create: a HEALPix histogram from the angular momentum of particles and angular distance plots.
    """


    def __init__(self, tag):
        """
        A constructor method for the class.
        :param tag: redshift directory.
        """
        # subgroup_numbers_all = np.load(data_path + tag + '/subgroup_numbers_all.npy')
        # subgroup_numbers_all = np.array([item for sublist in subgroup_numbers_all for item in sublist])
        #
        # glx_stellar_masses_all = np.load(data_path + tag + '/glx_stellar_masses_all.npy')
        # glx_stellar_masses_all = np.array([item for sublist in glx_stellar_masses_all for item in sublist])
        #
        # glx_disc_fractions_IT20_all = np.load(data_path + tag + '/glx_disc_fractions_IT20_all.npy')
        # glx_disc_fractions_IT20_all = np.array([item for sublist in glx_disc_fractions_IT20_all for item in sublist])
        #
        # glx_rotationals_over_dispersions_all = np.load(data_path + tag + '/glx_rotationals_over_dispersions_all.npy')
        # glx_rotationals_over_dispersions_all = np.array([item for sublist in glx_rotationals_over_dispersions_all for item in sublist])
        #
        # group_numbers_all = np.load(data_path + tag + '/group_numbers_all.npy')
        # group_numbers_all = np.array([item for sublist in group_numbers_all for item in sublist])
        #
        # morphology_mask, = np.array(np.where(
        #     (glx_stellar_masses_all > 1e10) & (glx_rotationals_over_dispersions_all > 1) & (glx_disc_fractions_IT20_all > 0.5) & (
        #         subgroup_numbers_all == 0)))
        # print(group_numbers_all[morphology_mask])
        # print(glx_disc_fractions_IT20_all[morphology_mask])

        group_numbers = [11, 37, 44, 158]
        regions = ['00', '00', '00', '00']

        # Generate the figure and define its parameters #
        figure = plt.figure(figsize=(20, 20))

        gs = gridspec.GridSpec(4, 4, wspace=0.4, hspace=0.4)
        axis00, axis01, axis02, axis03 = figure.add_subplot(gs[0, 0], projection='mollweide'), figure.add_subplot(gs[0, 1]), figure.add_subplot(
            gs[0, 2]), figure.add_subplot(gs[0, 3])
        axis10, axis11, axis12, axis13 = figure.add_subplot(gs[1, 0], projection='mollweide'), figure.add_subplot(gs[1, 1]), figure.add_subplot(
            gs[1, 2]), figure.add_subplot(gs[1, 3])
        axis20, axis21, axis22, axis23 = figure.add_subplot(gs[2, 0], projection='mollweide'), figure.add_subplot(gs[2, 1]), figure.add_subplot(
            gs[2, 2]), figure.add_subplot(gs[2, 3])
        axis30, axis31, axis32, axis33 = figure.add_subplot(gs[3, 0], projection='mollweide'), figure.add_subplot(gs[3, 1]), figure.add_subplot(
            gs[3, 2]), figure.add_subplot(gs[3, 3])

        for axis in [axis00, axis10, axis20, axis30]:
            axis.set_xlabel(r'$\mathrm{\alpha/\degree}$', size=20)
            axis.set_ylabel(r'$\mathrm{\delta/\degree}$', size=20)
            axis.set_yticklabels(['', '-60', '', '-30', '', '0', '', '30', '', '60', ''], size=20)
            axis.set_xticklabels(['', '-120', '', '-60', '', '0', '', '60', '', '120', ''], size=20)
        for axis in [axis01, axis11, axis21, axis31]:
            plot_tools.set_axis(axis, xlabel=r'$\mathrm{\Delta \theta/\degree}$', ylabel=r'$\mathrm{Particles\;per\;grid\;cell}$', aspect=None,
                                size=20)
        for axis in [axis02, axis12, axis22, axis32]:
            plot_tools.set_axis(axis, xlabel=r'$\mathrm{(Angular\;distance\;from\;\vec{J}_{gal})/\degree}$',
                                ylabel=r'$\mathrm{Particles\;per\;grid\;cell}$', aspect=None, size=20)
        for axis in [axis03, axis13, axis23, axis33]:
            plot_tools.set_axis(axis, xlabel=r'$\mathrm{\epsilon}$', ylabel=r'$\mathrm{f(\epsilon)}$', aspect=None, size=20)

        all_axes = [[axis00, axis01, axis02, axis03], [axis10, axis11, axis12, axis13], [axis20, axis21, axis22, axis23],
                    [axis30, axis31, axis32, axis33]]

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

                self.plot(axes, stellar_data_tmp, group_number)
                print('Plotted data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
                print('–––––––––––––––––––––––––––––––––––––––––––––')

        # Add text/title in each column #
        plt.text(0.25, 1.1, r'$\mathrm{D/T_{\Delta \theta<30\degree}}$', fontsize=30, transform=axis01.transAxes)
        plt.text(0.25, 1.1, r'$\mathrm{D/T_{\vec{J}_{b}=0}}$', fontsize=30, transform=axis02.transAxes)
        plt.text(0.25, 1.1, r'$\mathrm{D/T_{\epsilon>0.7}}$', fontsize=30, transform=axis03.transAxes)

        # Save and close the figure #
        plt.savefig(plots_path + 'SMD' + '-' + date + '.png', bbox_inches='tight')
        plt.close()
        print('Finished MultipleDecomposition for ' + tag + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')


    @staticmethod
    def plot(axes, stellar_data_tmp, group_number):
        """
        Plot a HEALPix histogram from the angular momentum of particles - an angular distance plot - a surface density plot / gri mock image - a
        circularity distribution.
        :param axes: set of axes
        :param stellar_data_tmp: from read_add_attributes.py.
        :param group_number: from read_add_attributes.py.
        :return: None
        """
        cos_angle_components = np.divide(
            np.sum(stellar_data_tmp['disc_stellar_angular_momentum'] * stellar_data_tmp['spheroid_stellar_angular_momentum']),
            np.linalg.norm(stellar_data_tmp['disc_stellar_angular_momentum']) * np.linalg.norm(
                stellar_data_tmp['spheroid_stellar_angular_momentum']))  # In radians.
        # print(np.degrees(np.arccos(cos_angle_components)))
        # Calculate the angular momentum for each particle and for the galaxy and the unit vector parallel to the galactic angular momentum vector #
        prc_angular_momentum = stellar_data_tmp['Mass'][:, np.newaxis] * np.cross(stellar_data_tmp['Coordinates'],
                                                                                  stellar_data_tmp['Velocity'])  # In Msun kpc km s^-1.
        glx_angular_momentum = np.sum(prc_angular_momentum, axis=0)
        glx_unit_vector = glx_angular_momentum / np.linalg.norm(glx_angular_momentum)

        # Rotate coordinates and velocities of stellar particles so the galactic angular momentum points along the x axis #
        coordinates, velocities, prc_unit_vector, glx_unit_vector = RotateCoordinates.rotate_X(stellar_data_tmp, glx_unit_vector)

        # Calculate the ra and dec of the (unit vector of) angular momentum for each particle #
        ra = np.degrees(np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0]))
        dec = np.degrees(np.arcsin(prc_unit_vector[:, 2]))

        # Plot a HEALPix histogram #
        nside = 2 ** 4  # Define the resolution of the grid (number of divisions along the side of a base-resolution grid cell).
        hp = HEALPix(nside=nside, order='ring')  # Initialise the HEALPix pixelisation class.
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
        axes[0].scatter(np.arctan2(glx_unit_vector[1], glx_unit_vector[0]), np.arcsin(glx_unit_vector[2]), s=100, color='black', marker='X',
                        facecolors='none', zorder=5)  # Position of the galactic angular momentum.
        axes[0].annotate(r'$\mathrm{Density\;maximum}$', xy=(lon_densest, lat_densest), xycoords='data', xytext=(0.2, 1.1),
                         textcoords='axes fraction', arrowprops=dict(arrowstyle='-', color='black', connectionstyle='arc3,rad=0'),
                         size=20)  # Position of the densest grid cell.

        # Sample a 360x180 grid in ra/dec #
        ra = np.linspace(-180.0, 180.0, num=360) * u.deg
        dec = np.linspace(-90.0, 90.0, num=180) * u.deg
        ra_grid, dec_grid = np.meshgrid(ra, dec)

        # Find density at each coordinate position #
        coordinate_index = hp.lonlat_to_healpix(ra_grid, dec_grid)
        density_map = densities[coordinate_index]

        # Display data on a 2D regular raster and create a pseudo-color plot #
        pcm = axes[0].pcolormesh(np.radians(ra), np.radians(dec), density_map, cmap='nipy_spectral_r')
        cbar = plt.colorbar(pcm, ax=axes[0], orientation='horizontal')
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label('$\mathrm{Particles\;per\;grid\;cell}$', size=20)

        # Calculate the disc mass fraction as the mass within 30 degrees from the densest grid cell #
        angular_theta_from_densest = np.arccos(
            np.sin(lat_densest) * np.sin(np.arcsin(prc_unit_vector[:, 2])) + np.cos(lat_densest) * np.cos(np.arcsin(prc_unit_vector[:, 2])) * np.cos(
                lon_densest - np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0])))  # In radians.
        disc_mask = np.where(angular_theta_from_densest < (np.pi / 6.0))
        disc_fraction_IT20 = np.sum(stellar_data_tmp['Mass'][disc_mask]) / np.sum(stellar_data_tmp['Mass'])

        # Calculate and plot the angular distance (spherical law of cosines) between the densest and all the other grid cells #
        angular_theta_from_densest = np.arccos(
            np.sin(lat_densest) * np.sin(np.radians(dec_grid.value)) + np.cos(lat_densest) * np.cos(np.radians(dec_grid.value)) * np.cos(
                lon_densest - np.radians(ra_grid.value)))  # In radians.

        axes[1].set_ylim(-5, 1.3 * max(density_map[density_map.nonzero()]))
        axes[1].scatter(angular_theta_from_densest[density_map.nonzero()] * (180.0 / np.pi), density_map[density_map.nonzero()], c='black',
                        s=5)  # In degrees.
        axes[1].axvline(x=30, c='blue', lw=3, linestyle='dashed')  # Vertical line at 30 degrees.
        axes[1].axvspan(0, 30, facecolor='0.2', alpha=0.5)  # Draw a vertical span.

        # Calculate and plot the distribution of orbital circularity #
        epsilon, stellar_masses = plot_tools.circularity(stellar_data_tmp, method='Thob')
        j, = np.where(epsilon < 0.0)
        k, = np.where((epsilon > 0.7) & (epsilon <= 1))
        l, = np.where((epsilon >= -1) & (epsilon <= 1))
        disc_fraction_00 = 1 - 2 * np.sum(stellar_masses[j]) / np.sum(stellar_masses[l])
        disc_fraction_07 = np.sum(stellar_masses[k]) / np.sum(stellar_masses[l])
        y_data, edges = np.histogram(epsilon, weights=stellar_masses / np.sum(stellar_masses), bins=50, range=[-1, 1])
        x_data = 0.5 * (edges[1:] + edges[:-1])
        y_data /= edges[1:] - edges[:-1]
        axes[3].set_xlim(-1.3, 1.3)
        axes[3].set_ylim(0, 1.4 * max(y_data))
        axes[3].plot(x_data, y_data, color='black')

        axes[3].fill_between(x_data[np.where(x_data <= 0.7)], y_data[np.where(x_data <= 0.7)], hatch='\\\\', facecolor='none', edgecolor='tab:red',
                             label=r'$\rm{Bulge}$')
        axes[3].fill_between(x_data[np.where(x_data >= 0.7)], y_data[np.where(x_data >= 0.7)], hatch='//', facecolor="none", edgecolor='tab:blue',
                             label=r'$\rm{Disc}$')

        # Calculate and plot the angular distance between the (unit vector of) the galactic angular momentum and all the other grid cells #
        position_of_X = np.vstack([np.arctan2(glx_unit_vector[1], glx_unit_vector[0]), np.arcsin(glx_unit_vector[2])]).T

        angular_theta_from_X = np.arccos(np.sin(position_of_X[0, 1]) * np.sin(np.radians(dec_grid.value)) + np.cos(position_of_X[0, 1]) * np.cos(
            np.radians(dec_grid.value)) * np.cos(position_of_X[0, 0] - np.radians(ra_grid.value)))  # In radians.
        axes[2].set_ylim(-5, 1.3 * max(density_map[density_map.nonzero()]))
        axes[2].scatter(angular_theta_from_X[density_map.nonzero()] * (180.0 / np.pi), density_map[density_map.nonzero()], c='black',
                        s=5)  # In degrees.
        axes[2].axvline(x=90, c='red', lw=3, linestyle='dashed')  # Vertical line at 30 degrees.
        axes[2].axvspan(90, 180, facecolor='0.2', alpha=0.5)  # Draw a vertical span.

        # Add text and create the legend #
        plt.text(-0.2, 1.1, str(group_number), color='red', fontsize=20, transform=axes[0].transAxes)
        plt.text(0.8, 0.9, r'$\mathrm{%.2f }$' % disc_fraction_IT20, fontsize=20, transform=axes[1].transAxes)
        plt.text(0.8, 0.9, r'$\mathrm{%.2f }$' % np.abs(disc_fraction_00), fontsize=20, transform=axes[2].transAxes)
        plt.text(0.8, 0.9, r'$\mathrm{%.2f }$' % disc_fraction_07, fontsize=20, transform=axes[3].transAxes)
        axes[3].legend(loc='upper left', fontsize=20, frameon=False)
        return None


if __name__ == '__main__':
    tag = '011_z004p770'
    data_path = '/cosma7/data/dp004/dc-irod1/FLARES/paper/data/'  # Path to save/load data.
    plots_path = '/cosma7/data/dp004/dc-irod1/FLARES/paper/plots/'  # Path to save plots.
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    x = SampleMultipleDecomposition(tag)
