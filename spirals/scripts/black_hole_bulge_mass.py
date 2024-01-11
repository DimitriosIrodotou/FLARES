# TODO ATM is uses argmax for BH_Mass, not np.sum of all BH_Mass
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
import matplotlib.path as mpath
from matplotlib.markers import MarkerStyle

style.use("classic")
obsHubble = 0.70  # [dimensionless]
plt.rcParams.update({'font.family': 'serif'})
date = time.strftime('%d_%m_%y_%H%M')  # Date.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


def plot_spiral_candidates():
    """
    Plot the black hole vs bulge mass relation of a given spiral candidate galaxy.
    :return: None
    """
    # Load the data. #
    all_group_numbers = np.load(data_path + 'group_numbers.npy')
    all_subgroup_numbers = np.load(data_path + 'subgroup_numbers.npy')
    stellar_data = np.load(data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' +
                           str(subgroup_number) + '.npy', allow_pickle=True)
    stellar_data = stellar_data.item()
    blackhole_data = np.load(data_path + 'blackhole_data_tmps/blackhole_data_tmp_' + str(group_number) + '_' +
                             str(subgroup_number) + '.npy', allow_pickle=True)
    blackhole_data = blackhole_data.item()

    glx_disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')

    # Normalise the disc fractions #
    chi = 0.5 * (1 - np.cos(np.pi / 6))
    glx_disc_fractions_IT20 = np.divide(1, 1 - chi) * (glx_disc_fractions_IT20 - chi)

    # Calculate the bulge mass. #
    mask, = np.where((all_group_numbers == float(group_number)) & (all_subgroup_numbers == float(subgroup_number)))
    bulge_mass = np.sum(stellar_data['Mass']) * (1 - glx_disc_fractions_IT20[mask])

    # Select the corresponding redshift axis. #
    if int(redshift) == 8:
        axis = axis00
    elif int(redshift) == 7:
        axis = axis01
    elif int(redshift) == 6:
        axis = axis10
    elif int(redshift) == 5:
        axis = axis11
    # Select the most massive BH in the galaxy. #
    blackhole_mass = blackhole_data['BH_Mass'][np.argmax(blackhole_data['BH_Mass'])]
    axis.scatter(bulge_mass, blackhole_mass, marker=get_hurricane(), s=1000, lw=5, edgecolors='k', facecolors='none')
    return None


def plot_barred_candidates():
    """
    Plot the black hole vs bulge mass relation of a given barred candidate galaxy.
    :return: None
    """
    # Load the data. #
    all_group_numbers = np.load(data_path + 'group_numbers.npy')
    all_subgroup_numbers = np.load(data_path + 'subgroup_numbers.npy')
    stellar_data = np.load(data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' +
                           str(subgroup_number) + '.npy', allow_pickle=True)
    stellar_data = stellar_data.item()
    blackhole_data = np.load(data_path + 'blackhole_data_tmps/blackhole_data_tmp_' + str(group_number) + '_' +
                             str(subgroup_number) + '.npy', allow_pickle=True)
    blackhole_data = blackhole_data.item()

    glx_disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')

    # Normalise the disc fractions #
    chi = 0.5 * (1 - np.cos(np.pi / 6))
    glx_disc_fractions_IT20 = np.divide(1, 1 - chi) * (glx_disc_fractions_IT20 - chi)

    # Calculate the bulge mass. #
    mask, = np.where((all_group_numbers == float(group_number)) & (all_subgroup_numbers == float(subgroup_number)))
    bulge_mass = np.sum(stellar_data['Mass']) * (1 - glx_disc_fractions_IT20[mask])

    # Select the corresponding redshift axis. #
    if int(redshift) == 8:
        axis = axis00
    elif int(redshift) == 7:
        axis = axis01
    elif int(redshift) == 6:
        axis = axis10
    elif int(redshift) == 5:
        axis = axis11

    # Select the most massive BH in the galaxy. #
    blackhole_mass = blackhole_data['BH_Mass'][np.argmax(blackhole_data['BH_Mass'])]
    bar = MarkerStyle("|")
    bar._transform.rotate_deg(90)
    axis.scatter(bulge_mass, blackhole_mass, marker=bar, s=1000, lw=5, color='k')
    return None


def plot_disc_dominated_galaxies():
    """
    Plot the black hole vs bulge mass relation of a disc-dominated candidate galaxy.
    :return: None
    """
    # Load the data. #
    try:
        all_group_numbers = np.load(data_path + 'group_numbers.npy')
    except FileNotFoundError:
        return None
    if len(all_group_numbers) == 0:
        return None

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
        blackhole_data = np.load(data_path + 'blackhole_data_tmps/blackhole_data_tmp_' + str(group_number) + '_' +
                                 str(subgroup_number) + '.npy', allow_pickle=True)
        stellar_data, blackhole_data = stellar_data.item(), blackhole_data.item()

        mask, = np.where((all_group_numbers == float(group_number)) & (all_subgroup_numbers == float(subgroup_number)))
        glx_disc_fraction_IT20 = glx_disc_fractions_IT20[mask]

        bulge_mass = np.sum(stellar_data['Mass']) * (1 - glx_disc_fraction_IT20)
        # Select the most massive BH in the galaxy. #
        if len(blackhole_data['BH_Mass']) > 0:
            blackhole_mass = blackhole_data['BH_Mass'][np.argmax(blackhole_data['BH_Mass'])]
            axis.scatter(bulge_mass, blackhole_mass, s=20, color='tab:blue')
        else:
            continue
    return None


def plot_bulge_dominated_galaxies():
    """
    Plot the black hole vs bulge mass relation of a bulge-dominated candidate galaxy.
    :return: None
    """
    # Load the data. #
    try:
        all_group_numbers = np.load(data_path + 'group_numbers.npy')
    except FileNotFoundError:
        return None
    if len(all_group_numbers) == 0:
        return None

    all_subgroup_numbers = np.load(data_path + 'subgroup_numbers.npy')
    glx_disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')

    # Normalise the disc fractions #
    chi = 0.5 * (1 - np.cos(np.pi / 6))
    glx_disc_fractions_IT20 = np.divide(1, 1 - chi) * (glx_disc_fractions_IT20 - chi)
    mask_disc_fraction, = np.where(glx_disc_fractions_IT20 < 0.5)

    for group_number, subgroup_number in zip(all_group_numbers[mask_disc_fraction],
                                             all_subgroup_numbers[mask_disc_fraction]):
        stellar_data = np.load(
            data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy',
            allow_pickle=True)
        blackhole_data = np.load(data_path + 'blackhole_data_tmps/blackhole_data_tmp_' + str(group_number) + '_' +
                                 str(subgroup_number) + '.npy', allow_pickle=True)
        stellar_data, blackhole_data = stellar_data.item(), blackhole_data.item()

        mask, = np.where((all_group_numbers == float(group_number)) & (all_subgroup_numbers == float(subgroup_number)))
        glx_disc_fraction_IT20 = glx_disc_fractions_IT20[mask]

        bulge_mass = np.sum(stellar_data['Mass']) * (1 - glx_disc_fraction_IT20)
        # Select the most massive BH in the galaxy. #
        if len(blackhole_data['BH_Mass']) > 0:
            blackhole_mass = blackhole_data['BH_Mass'][np.argmax(blackhole_data['BH_Mass'])]
            axis.scatter(bulge_mass, blackhole_mass, s=20, color='tab:red')
        else:
            continue
    return None


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
    figure = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(nrows=2, ncols=2)
    axis00, axis01 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1])
    axis10, axis11 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1])
    axes = [axis00, axis01, axis10, axis11]
    for axis in axes:
        plot_tools.set_axes(axis, xlim=[1e9, 1e11], ylim=[1e5, 1e9], xscale='log', yscale='log',
                            xlabel='$M_\mathrm{bulge} /\mathrm{M_\odot}$', ylabel=r'$M_\bullet /\mathrm{M_\odot}$',
                            aspect=None, which='major', size=30)

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
            plot_bulge_dominated_galaxies()

    # Save and close  the figure. #
    plt.savefig('/cosma7/data/dp004/dc-irod1/FLARES/spirals/plots/black_hole_bulge_mass.png', bbox_inches='tight')
    plt.close()
