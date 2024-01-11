import re
import time
import warnings
import matplotlib
import plot_tools

matplotlib.use('Agg')

import numpy as np
import matplotlib.cbook
import astropy.units as u
from matplotlib import gridspec
import matplotlib.pyplot as plt
from scipy.special import gamma
import matplotlib.path as mpath
from astropy.constants import G
from scipy.optimize import curve_fit
import matplotlib.style as style
from plot_tools import RotateCoordinates
from matplotlib.markers import MarkerStyle

G = G.to(u.kpc * u.Msun ** -1 * u.km ** 2 * u.s ** -2).value  # In kpc Msun^-1 km^2 s^-2.

style.use("classic")
obsHubble = 0.70  # [dimensionless]
plt.rcParams.update({'font.family': 'serif'})
date = time.strftime('%d_%m_%y_%H%M')  # Date.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


def profile_fitting(data_tmp):
    """
    Calculate the Sersic index.
    :param data_tmp: Dictionary of data
    :return: R_d, disk_mass
    """

    def exponential_profile(r, I_0d, R_d):
        """
        Calculate an exponential profile.
        :param r: radius
        :param I_0d: Disc central intensity.
        :param R_d: Disc scale length.
        :return: I_0d * np.exp(-r / R_d)
        """
        return I_0d * np.exp(-r / R_d)

    # Rotate coordinates and velocities of stellar particles wrt galactic angular momentum #
    coordinates, velocities, prc_angular_momentum, glx_angular_momentum = RotateCoordinates.rotate_Jz(data_tmp)

    cylindrical_distance = np.sqrt(coordinates[:, 0] ** 2 + coordinates[:, 1] ** 2)  # Radius of each particle.
    vertical_mask, = np.where(abs(coordinates[:, 2]) < 1)  # Vertical cut in kpc.

    mass, edges = np.histogram(cylindrical_distance[vertical_mask], bins=10, range=(0, 5),
                               weights=data_tmp['Mass'][vertical_mask])
    centers = 0.5 * (edges[1:] + edges[:-1])
    surface = np.pi * (edges[1:] ** 2 - edges[:-1] ** 1)
    sden = mass / surface
    try:
        popt, pcov = curve_fit(exponential_profile, centers, sden, sigma=0.1 * sden,
                               p0=[sden[0], 1])  # p0 = [I_0d, R_d]
        # Calculate galactic attributes #
        I_0d, R_d = popt[0], popt[1]
        disk_mass = 2.0 * np.pi * I_0d * R_d ** 2

    except RuntimeError:
        R_d, disk_mass = 0, 0
        print('Could not fit an exponential profile')

    return R_d, disk_mass


def plot_spiral_candidates():
    """
    Plot stability criteria vs disc-to-total mass ratio of a given spiral candidate galaxy.
    :return: None
    """
    # Load the data. #
    glx_rotationals = np.load(data_path + 'glx_rotationals.npy')
    all_group_numbers = np.load(data_path + 'group_numbers.npy')
    all_subgroup_numbers = np.load(data_path + 'subgroup_numbers.npy')

    stellar_data = np.load(data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' +
                           str(subgroup_number) + '.npy', allow_pickle=True)
    gaseous_data = np.load(data_path + 'gaseous_data_tmps/gaseous_data_tmp_' + str(group_number) + '_' +
                           str(subgroup_number) + '.npy', allow_pickle=True)
    stellar_data, gaseous_data = stellar_data.item(), gaseous_data.item()
    glx_disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')

    # Normalise the disc fractions #
    chi = 0.5 * (1 - np.cos(np.pi / 6))
    glx_disc_fractions_IT20 = np.divide(1, 1 - chi) * (glx_disc_fractions_IT20 - chi)

    # Calculate the D/T ratio. #
    mask, = np.where((all_group_numbers == float(group_number)) & (all_subgroup_numbers == float(subgroup_number)))

    # Select the corresponding redshift axis. #
    if int(redshift) == 8:
        axis = axis00
    elif int(redshift) == 7:
        axis = axis01
    elif int(redshift) == 6:
        axis = axis02
    elif int(redshift) == 5:
        axis = axis03

    # Calculate the stellar and gaseous disc scale lengths by fitting exponential profile. #
    stellar_scale_length, stellar_disc_mass = profile_fitting(stellar_data)
    gaseous_scale_length, gaseous_disc_mass = profile_fitting(gaseous_data)
    # Only use galaxies whose scale length is not equal to the initial guess value of 1 kpc. #
    if stellar_scale_length != 1 and gaseous_scale_length != 1:
        # Efstathiou, Lake, Negroponte 1982 stability criterion. #
        ELN_criterion = G * stellar_disc_mass / (glx_rotationals[mask] ** 2 * stellar_scale_length)
        axis.scatter(ELN_criterion, glx_disc_fractions_IT20[mask], marker=get_hurricane(), s=1000, lw=5,
                     edgecolors='tab:blue', facecolors='none')

        # Irodotou, Thomas, Henriques et al. 2019 stability criterion. #
        epsilon_stars = 1.1 * np.sqrt(G * stellar_disc_mass / (glx_rotationals[mask] ** 2 * stellar_scale_length))
        epsilon_gas = 0.9 * np.sqrt(G * gaseous_disc_mass / (glx_rotationals[mask] ** 2 * gaseous_scale_length))
        ITH_criterion = (stellar_disc_mass * epsilon_stars + gaseous_disc_mass * epsilon_gas) \
                        / (stellar_disc_mass + gaseous_disc_mass)
        axis.scatter(ITH_criterion, glx_disc_fractions_IT20[mask], marker=get_hurricane(), s=1000, lw=5,
                     edgecolors='tab:cyan', facecolors='none')

    return None


def plot_barred_candidates():
    """
    Plot stability criteria vs disc-to-total mass ratio of a given barred candidate galaxy.
    :return: None
    """
    # Load the data. #
    glx_rotationals = np.load(data_path + 'glx_rotationals.npy')
    all_group_numbers = np.load(data_path + 'group_numbers.npy')
    all_subgroup_numbers = np.load(data_path + 'subgroup_numbers.npy')

    stellar_data = np.load(data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' +
                           str(subgroup_number) + '.npy', allow_pickle=True)
    gaseous_data = np.load(data_path + 'gaseous_data_tmps/gaseous_data_tmp_' + str(group_number) + '_' +
                           str(subgroup_number) + '.npy', allow_pickle=True)
    stellar_data, gaseous_data = stellar_data.item(), gaseous_data.item()
    glx_disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')

    # Normalise the disc fractions #
    chi = 0.5 * (1 - np.cos(np.pi / 6))
    glx_disc_fractions_IT20 = np.divide(1, 1 - chi) * (glx_disc_fractions_IT20 - chi)

    # Calculate the bulge mass. #
    mask, = np.where((all_group_numbers == float(group_number)) & (all_subgroup_numbers == float(subgroup_number)))

    # Select the corresponding redshift axis. #
    if int(redshift) == 8:
        axis = axis00
    elif int(redshift) == 7:
        axis = axis01
    elif int(redshift) == 6:
        axis = axis02
    elif int(redshift) == 5:
        axis = axis03

    bar = MarkerStyle("|")
    bar._transform.rotate_deg(90)

    # Calculate the stellar and gaseous disc scale lengths by fitting exponential profile. #
    stellar_scale_length, stellar_disc_mass = profile_fitting(stellar_data)
    gaseous_scale_length, gaseous_disc_mass = profile_fitting(gaseous_data)
    # Only use galaxies whose scale length is not equal to the initial guess value of 1 kpc. #
    if stellar_scale_length != 1 and gaseous_scale_length != 1:
        # Efstathiou, Lake, Negroponte 1982 stability criterion. #
        ELN_criterion = G * stellar_disc_mass / (glx_rotationals[mask] ** 2 * stellar_scale_length)
        axis.scatter(ELN_criterion, glx_disc_fractions_IT20[mask], marker=bar, s=1000, lw=5, color='tab:red')

        # Irodotou, Thomas, Henriques et al. 2019 stability criterion. #
        epsilon_stars = 1.1 * np.sqrt(G * stellar_disc_mass / (glx_rotationals[mask] ** 2 * stellar_scale_length))
        epsilon_gas = 0.9 * np.sqrt(G * gaseous_disc_mass / (glx_rotationals[mask] ** 2 * gaseous_scale_length))
        ITH_criterion = (stellar_disc_mass * epsilon_stars + gaseous_disc_mass * epsilon_gas) \
                        / (stellar_disc_mass + gaseous_disc_mass)
        axis.scatter(ITH_criterion, glx_disc_fractions_IT20[mask], marker=bar, s=1000, lw=5, color='tab:orange')
    return None


def plot_disc_dominated_galaxies():
    """
    Plot stability criteria vs disc-to-total mass ratio of a disc-dominated candidate galaxy.
    :return: None
    """
    # Load the data. #
    try:
        all_group_numbers = np.load(data_path + 'group_numbers.npy')
    except FileNotFoundError:
        return None
    if len(all_group_numbers) == 0:
        return None

    glx_rotationals = np.load(data_path + 'glx_rotationals.npy')
    all_subgroup_numbers = np.load(data_path + 'subgroup_numbers.npy')
    glx_disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')

    # Normalise the disc fractions #
    chi = 0.5 * (1 - np.cos(np.pi / 6))
    glx_disc_fractions_IT20 = np.divide(1, 1 - chi) * (glx_disc_fractions_IT20 - chi)
    mask_disc_fraction, = np.where(glx_disc_fractions_IT20 >= 0.5)

    # Add text. #
    axis.set_title('$z = $' + str(redshift), color='k', fontsize=30)

    for group_number, subgroup_number in zip(all_group_numbers[mask_disc_fraction],
                                             all_subgroup_numbers[mask_disc_fraction]):
        stellar_data = np.load(
            data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy',
            allow_pickle=True)
        gaseous_data = np.load(
            data_path + 'gaseous_data_tmps/gaseous_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy',
            allow_pickle=True)
        stellar_data, gaseous_data = stellar_data.item(), gaseous_data.item()

        mask, = np.where((all_group_numbers == float(group_number)) & (all_subgroup_numbers == float(subgroup_number)))

        # Calculate the stellar and gaseous disc scale lengths by fitting exponential profile. #
        stellar_scale_length, stellar_disc_mass = profile_fitting(stellar_data)
        gaseous_scale_length, gaseous_disc_mass = profile_fitting(gaseous_data)
        # Only use galaxies whose scale length is not equal to the initial guess value of 1 kpc. #
        if stellar_scale_length != 1 and gaseous_scale_length != 1:
            # Efstathiou, Lake, Negroponte 1982 stability criterion. #
            ELN_criterion = G * stellar_disc_mass / (glx_rotationals[mask] ** 2 * stellar_scale_length)
            axis.scatter(ELN_criterion, glx_disc_fractions_IT20[mask], s=20, color='k', zorder=-5)

            # Irodotou, Thomas, Henriques et al. 2019 stability criterion. #
            epsilon_stars = 1.1 * np.sqrt(G * stellar_disc_mass / (glx_rotationals[mask] ** 2 * stellar_scale_length))
            epsilon_gas = 0.9 * np.sqrt(G * gaseous_disc_mass / (glx_rotationals[mask] ** 2 * gaseous_scale_length))
            ITH_criterion = (stellar_disc_mass * epsilon_stars + gaseous_disc_mass * epsilon_gas) \
                            / (stellar_disc_mass + gaseous_disc_mass)
            axis.scatter(ITH_criterion, glx_disc_fractions_IT20[mask], s=20, color='tab:grey', zorder=-5)


def get_hurricane():
    # Use custom points to define a half-spiral. #
    u = np.array([[2.5, 7.5], [0.5, 7], [-1.5, 5.5], [-2.5, 3], [-2.5, 0], [-2, -2], [0, -3], [2.5, -2]])
    codes = [1] + [2] * (len(u) - 2) + [2]
    u = np.append(u, -u[::-1], axis=0)
    codes += codes
    return mpath.Path(u, codes, closed=False)


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
    figure = plt.figure(figsize=(40, 10))
    gs = gridspec.GridSpec(nrows=1, ncols=4)
    axis00, axis01, axis02, axis03 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2]), \
                                     plt.subplot(gs[0, 3])
    axes = [axis00, axis01, axis02, axis03]
    for axis in axes:
        plot_tools.set_axes(axis, xlim=[-0.1, 4], ylim=[0.45, 0.9],
                            xlabel=r'$M_\bigstar /\mathrm{M}_\odot$', ylabel=r'$D/T$', which='major',
                            size=30)
        if axis != axis00:
            axis.set_ylabel(''), axis.set_yticklabels([])

    # Plot the spiral candidates. #
    redshifts, tags, regions, group_numbers, subgroup_numbers = [], [], [], [], []
    for candidate in spiral_candidates:
        redshifts.append(re.split('z00|p', candidate)[1])
        tags.append(re.split('_', candidate)[0] + '_' + re.split('_', candidate)[1])
        regions.append(re.split('_', candidate)[2])
        group_numbers.append(re.split('_', candidate)[3]), subgroup_numbers.append(re.split('_', candidate)[4])
    # Loop over candidates and plot them. #
    for redshift, tag, region, group_number, subgroup_number in zip(redshifts, tags, regions, group_numbers,
                                                                    subgroup_numbers):
        data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/JWST/' + region + '/' + tag + '/'
        plot_spiral_candidates()

    # Plot the barred candidates. #
    redshifts, tags, regions, group_numbers, subgroup_numbers = [], [], [], [], []
    for candidate in barred_candidates:
        redshifts.append(re.split('z00|p', candidate)[1])
        tags.append(re.split('_', candidate)[0] + '_' + re.split('_', candidate)[1])
        regions.append(re.split('_', candidate)[2])
        group_numbers.append(re.split('_', candidate)[3]), subgroup_numbers.append(re.split('_', candidate)[4])
    # Loop over candidates and plot them. #
    for redshift, tag, region, group_number, subgroup_number in zip(redshifts, tags, regions, group_numbers,
                                                                    subgroup_numbers):
        data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/JWST/' + region + '/' + tag + '/'
        plot_barred_candidates()

    # Plot the disc dominated galaxies. #
    redshifts = [8, 7, 6, 5]
    tags = ['007_z008p000', '008_z007p000', '009_z006p000', '010_z005p000']
    for axis, tag, redshift in zip(axes, tags, redshifts):
        if redshift <= 2.8:
            softening = 0.000474390 / 0.6777 * 1e3  # In pkpc.
        else:
            softening = 0.001802390 / (0.6777 * (1 + redshift)) * 1e3  # In pkpc.

        for region in ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15',
                       '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32',
                       '33', '34', '35', '36', '37', '38', '39']:
            # Path to load data and to save plots.
            data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/JWST/' + region + '/' + tag + '/'
            plots_path = '/cosma7/data/dp004/dc-irod1/FLARES/spirals/plots/' + region + '/' + tag + '/'
            plot_disc_dominated_galaxies()

    # Save and close  the figure. #
    plt.savefig('/cosma7/data/dp004/dc-irod1/FLARES/spirals/plots/stability_DoverT.png', bbox_inches='tight')
    plt.close()
