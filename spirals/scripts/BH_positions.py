import time
import plot_tools
import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
import eagle_IO.eagle_IO.eagle_IO as E
from astropy.cosmology import z_at_value

from flares import flares
from matplotlib import gridspec

start_time = time.time()
date = time.strftime('%d_%m_%y_%H%M')

# Load data #
fl = flares.flares(fname='/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5')
gr_number = fl.load_dataset('GroupNumber', arr_type='Galaxy')
mstar = fl.load_dataset('Mstar_aperture/30', arr_type='Galaxy')
sg_number = fl.load_dataset('SubGroupNumber', arr_type='Galaxy')
sfr = fl.load_dataset('SFR_aperture/30/inst', arr_type='Galaxy')

# Define the redshift #
z = 5.00
tag = '010_z005p000'
if z <= 2.8:
    softening = 0.000474390 / 0.6777 * 1e3  # In pkpc.
else:
    softening = 0.001802390 / (0.6777 * (1 + z)) * 1e3  # In pkpc.

# Loop over the 40 haloes #
haloes = ['03']
for halo in haloes:
    # for halo in fl.halos:

    # Load star formation rate and stellar masses and convert their units #
    _sfr, _mstar = sfr[halo][tag], mstar[halo][tag] * 1e10
    _sfr[_sfr <= 0], _mstar[_mstar <= 0] = 1e-10, 1e4
    _sfr, _mstar = np.log10(_sfr), np.log10(_mstar)
    ssfr = _sfr - _mstar
    ssfr += 9  # convert to 1/Gyr

    # Select galaxies that are massive and quenched #
    centrals, satellites = (sg_number[halo][tag] == 0), (sg_number[halo][tag] > 0)
    idxs_centrals = np.where((ssfr < -1) & (_mstar > 9.7) & centrals)[0]
    idxs_satellites = np.where((ssfr < -1) & (_mstar > 9.7) & satellites)[0]
    if len(idxs_centrals) == 0 and len(idxs_satellites) == 0:
        continue
    else:
        print("Halo:", halo, "Galaxies:", len(_mstar), "Centrals/satellites:", len(np.where(centrals)[0]), "/",
              len(np.where(satellites)[0]), "Quenched centrals/satellites:", len(idxs_centrals), "/",
              len(idxs_satellites))

    # Extract stellar, black hole and halo data #
    sim = "%s/GEAGLE_%s/data" % (fl.directory, halo)
    gas_data, stellar_data, blackhole_data, subhalo_data = {}, {}, {}, {}
    for attribute in ['Coordinates', 'Mass', 'GroupNumber', 'SubGroupNumber']:
        gas_data[attribute] = E.read_array('PARTDATA', sim, tag, '/PartType0/' + attribute, numThreads=8)
    for attribute in ['Coordinates', 'Mass', 'GroupNumber', 'SubGroupNumber']:
        stellar_data[attribute] = E.read_array('PARTDATA', sim, tag, '/PartType4/' + attribute, numThreads=8)
    for attribute in ['BH_Mass', 'Coordinates', 'GroupNumber', 'BH_MostMassiveProgenitorID', 'ParticleIDs',
                      'SmoothingLength', 'SubGroupNumber', 'Velocity']:
        blackhole_data[attribute] = E.read_array('PARTDATA', sim, tag, '/PartType5/' + attribute, numThreads=8)
    for attribute in ['CentreOfPotential', 'GroupNumber', 'SubGroupNumber']:
        subhalo_data[attribute] = E.read_array('SUBFIND', sim, tag, '/Subhalo/' + attribute, numThreads=8)
    blackhole_details = pd.read_csv(
        '/cosma7/data/dp004/dc-love2/codes/flares_passive/analysis/data/blackhole_details_h%s.csv' % halo)

    gas_data['Mass'] *= u.g.to(u.Msun)
    stellar_data['Mass'] *= u.g.to(u.Msun)
    gas_data['Coordinates'] *= u.cm.to(u.kpc)
    blackhole_data['BH_Mass'] *= u.g.to(u.Msun)
    blackhole_data['Velocity'] *= u.cm.to(u.km)  # s^-1.
    stellar_data['Coordinates'] *= u.cm.to(u.kpc)
    blackhole_data['Coordinates'] *= u.cm.to(u.kpc)
    blackhole_data['SmoothingLength'] *= u.cm.to(u.kpc)
    subhalo_data['CentreOfPotential'] *= u.cm.to(u.kpc)

    # Loop over all quenched and central galaxies in the particular halo #
    centrals = [104]
    for _idx in centrals:
        # for _idx in idxs_centrals:

        # Create the plot
        figure = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(2, 2, hspace=0.5, wspace=0.5)
        axis00, axis01, = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1])
        axis10, axis11 = plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1])
        plot_tools.set_axes(axis00, xlim=[-40, 40], ylim=[-40, 40], xticks=[np.arange(-40, 41, 20)],
                            yticks=[np.arange(-40, 41, 20)], xlabel=r'$\mathrm{x/kpc}$',
                            ylabel=r'$\mathrm{y/kpc}$', size=30)
        plot_tools.set_axes(axis01, xlim=[5, max(np.log10(blackhole_data['BH_Mass']))],
                            ylim=[0, 40],
                            xticks=[np.arange(5, max(np.log10(blackhole_data['BH_Mass'])) + 1, 1)],
                            xlabel=r'$\mathrm{log_{10}(M_\bullet/M_\odot)}$', ylabel=r'$\mathrm{N}$', size=30)
        plot_tools.set_axes(axis10, xlim=[0, 50], ylim=[0, 10], xlabel=r'$\mathrm{(r_\bullet - r_{CoM})/kpc}$',
                            ylabel=r'$\mathrm{N}$', size=30)
        plot_tools.set_axes(axis11, xlim=[0, fl.cosmo.age(z).value], ylim=[5, max(np.log10(blackhole_data['BH_Mass']))],
                            xticks=[np.arange(0, fl.cosmo.age(z).value, 0.5)],
                            yticks=[np.arange(5, max(np.log10(blackhole_data['BH_Mass'])) + 1, 1)],
                            xlabel=r'$\mathrm{Age/Gyr}$', ylabel=r'$\mathrm{M_\bullet/M_\odot}$', size=30)

        # Get masks for each specific galaxy #
        gas_mask = np.where((gas_data['GroupNumber'] == gr_number[halo][tag][_idx]) & (
                gas_data['SubGroupNumber'] == sg_number[halo][tag][_idx]))
        stellar_mask = np.where((stellar_data['GroupNumber'] == gr_number[halo][tag][_idx]) & (
                stellar_data['SubGroupNumber'] == sg_number[halo][tag][_idx]))
        blackhole_mask = np.where((blackhole_data['GroupNumber'] == gr_number[halo][tag][
            _idx]))  # & (blackhole_data['SubGroupNumber'] == sg_number[halo][tag][_idx])) TODO Satellites are commented out
        halo_mask, = np.where((subhalo_data['GroupNumber'] == gr_number[halo][tag][_idx]) & (
                subhalo_data['SubGroupNumber'] == sg_number[halo][tag][_idx]))

        # Mask the temporary dictionary for each galaxy #
        gas_data_tmp, stellar_data_tmp, blackhole_data_tmp, subhalo_data_tmp = {}, {}, {}, {}
        for attribute in gas_data.keys():
            gas_data_tmp[attribute] = np.copy(gas_data[attribute])[gas_mask]
        for attribute in stellar_data.keys():
            stellar_data_tmp[attribute] = np.copy(stellar_data[attribute])[stellar_mask]
        for attribute in blackhole_data.keys():
            blackhole_data_tmp[attribute] = np.copy(blackhole_data[attribute])[blackhole_mask]
        for attribute in subhalo_data.keys():
            subhalo_data_tmp[attribute] = np.copy(subhalo_data[attribute])[halo_mask]

        # Normalise coordinates #
        for data in [gas_data_tmp, stellar_data_tmp, blackhole_data_tmp]:
            data['Coordinates'] = data['Coordinates'] - subhalo_data_tmp['CentreOfPotential']

        # Plot the x-y spatial positions of stellar and black hole particles #
        axis00.scatter(gas_data_tmp['Coordinates'][:, 0], gas_data_tmp['Coordinates'][:, 1], s=1, color='grey')
        axis00.scatter(stellar_data_tmp['Coordinates'][:, 0], stellar_data_tmp['Coordinates'][:, 1], s=10, color='k')
        axis00.scatter(blackhole_data_tmp['Coordinates'][:, 0], blackhole_data_tmp['Coordinates'][:, 1], s=10,
                       color='tab:red')

        # Plot circles with each black hole's smoothing and 3*softening length #
        for i in range(len(blackhole_data_tmp['ParticleIDs'])):
            bh_sml = plt.Circle((blackhole_data_tmp['Coordinates'][i, 0], blackhole_data_tmp['Coordinates'][i, 1]),
                                blackhole_data_tmp['SmoothingLength'][i], color='red', fill=False)
            sl = plt.Circle((blackhole_data_tmp['Coordinates'][i, 0], blackhole_data_tmp['Coordinates'][i, 1]),
                            3 * softening, color='blue', fill=False)
            axis00.add_patch(bh_sml)
            axis00.add_patch(sl)

        # Plot a 30 kpc aperture circle #
        aperture = plt.Circle((0, 0), 30, color='green', lw=4, fill=False)
        axis00.add_patch(aperture)

        # Plot the black hole mass histogram #
        axis01.hist(np.log10(blackhole_data_tmp['BH_Mass']),
                    bins=10 * int(max(axis01.get_xticks()) - min(axis01.get_xticks())),
                    range=(int(min(axis01.get_xticks())), int(max(axis01.get_xticks()))), color='k')

        axis01.text(0.05, 0.94, '$\mathrm{log_{10}}(M_{*} \,/\, M_{\odot}) = %.2f$' % _mstar[_idx],
                    transform=axis01.transAxes)
        axis01.text(0.05, 0.86, '$\mathrm{sSFR \,/\, Gyr} = %.2f$' % (10 ** ssfr[_idx]), transform=axis01.transAxes)
        axis01.text(0.05, 0.78, '$\mathrm{log_{10}}(M_{BH} \,/\, M_{\odot}) = %.2f$' % (
            np.log10(blackhole_data_tmp['BH_Mass'][np.argmax(blackhole_data_tmp['BH_Mass'])])),
                    transform=axis01.transAxes)

        # Loop over all black holes and calculate relative distances #
        # positions = np.linalg.norm(blackhole_data_tmp['Coordinates'], axis=1)
        # distances = np.zeros([len(blackhole_data_tmp['ParticleIDs']), len(blackhole_data_tmp['ParticleIDs'])])
        # for i in np.arange(0, len(blackhole_data_tmp['ParticleIDs']), 1):
        #     for j in np.arange(i + 1, len(blackhole_data_tmp['ParticleIDs']), 1):
        #         distances[i, j] = np.abs(positions[i] - positions[j])
        #
        # axis10.hist(distances[distances != 0.0], bins=int(np.floor(50 / (3 * softening))),
        #             range=(int(min(axis10.get_xticks())), int(max(axis10.get_xticks()))), color='k')

        axis10.hist(np.linalg.norm(blackhole_data_tmp['Coordinates'], axis=1), bins=int(np.floor(50 / (3 * softening))),
                    range=(int(min(axis10.get_xticks())), int(max(axis10.get_xticks()))), color='k')

        # Loop over all black holes and calculate relative velocities #
        # velocities = np.linalg.norm(blackhole_data_tmp['Velocity'], axis=1)
        # relative_velocities = np.zeros([len(blackhole_data_tmp['ParticleIDs']), len(blackhole_data_tmp['ParticleIDs'])])
        # for i in np.arange(0, len(blackhole_data_tmp['ParticleIDs']), 1):
        #     for j in np.arange(i + 1, len(blackhole_data_tmp['ParticleIDs']), 1):
        #         relative_velocities[i, j] = np.abs(velocities[i] - velocities[j])
        #
        # axis11.hist(relative_velocities[relative_velocities != 0.0], bins=len(blackhole_data_tmp['BH_Mass']),
        #             range=(int(min(axis11.get_xticks())), int(max(axis11.get_xticks()))), color='k')

        # Loop over all black holes in the galaxy and plot their black hole mass #
        for id in blackhole_data_tmp['ParticleIDs']:
            if id == blackhole_data_tmp['ParticleIDs'][np.argmax(blackhole_data_tmp['BH_Mass'])]:
                color = 'tab:red'
            else:
                color = 'tab:blue'
            mask = (blackhole_details['PID'] == id)
            bh_history = blackhole_details.loc[mask].sort_values('Time')
            bh_history['z'] = (1 / bh_history['Time']) - 1
            bh_history['Age'] = fl.cosmo.age(bh_history['z'])
            axis11.plot(bh_history['Age'], np.log10(bh_history['BH_Subgrid_Mass'] * 1e10), color=color)

        plots_path = '/cosma7/data/dp004/dc-irod1/FLARES/Lovell_21/plots/'  # Path to save plots.
        plt.savefig(plots_path + 'BH_positions' + '-' + date + '-' + str(halo) + '-' + str(_idx) + '.png',
                    bbox_inches='tight')
        plt.close()
print('Finished main.py in %.4s s' % (time.time() - start_time))
