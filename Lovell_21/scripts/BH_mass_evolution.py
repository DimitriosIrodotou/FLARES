import time
import plot_tools
import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
import eagle_IO.eagle_IO.eagle_IO as E

from flares import flares

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
fl.halos = ['03']
for halo in fl.halos:
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
    blackhole_data, subhalo_data = {}, {}
    for attribute in ['BH_Mass', 'Coordinates', 'GroupNumber', 'BH_MostMassiveProgenitorID', 'ParticleIDs',
                      'SmoothingLength', 'SubGroupNumber', 'Velocity']:
        blackhole_data[attribute] = E.read_array('PARTDATA', sim, tag, '/PartType5/' + attribute, numThreads=8)
    for attribute in ['CentreOfPotential', 'GroupNumber', 'SubGroupNumber']:
        subhalo_data[attribute] = E.read_array('SUBFIND', sim, tag, '/Subhalo/' + attribute, numThreads=8)
    blackhole_details = pd.read_csv(
        '/cosma7/data/dp004/dc-love2/codes/flares_passive/analysis/data/blackhole_details_h%s.csv' % halo)

    blackhole_data['BH_Mass'] *= u.g.to(u.Msun)
    blackhole_data['Velocity'] *= u.cm.to(u.km)  # s^-1.
    blackhole_data['Coordinates'] *= u.cm.to(u.kpc)
    blackhole_data['SmoothingLength'] *= u.cm.to(u.kpc)
    subhalo_data['CentreOfPotential'] *= u.cm.to(u.kpc)

    # Loop over all quenched and central galaxies in the particular halo #
    idxs_centrals = [104]
    for _idx in idxs_centrals:

        # Create the plot
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        plot_tools.set_axes(axis, xlim=[0, fl.cosmo.age(z).value], ylim=[5, max(np.log10(blackhole_data['BH_Mass']))],
                            xticks=[np.arange(0, fl.cosmo.age(z).value, 0.5)],
                            yticks=[np.arange(5, max(np.log10(blackhole_data['BH_Mass'])) + 1, 1)],
                            xlabel=r'$\mathrm{Age/Gyr}$', ylabel=r'$\mathrm{M_\bullet/M_\odot}$', size=30)

        # Get masks for each specific galaxy #
        blackhole_mask = np.where((blackhole_data['GroupNumber'] == gr_number[halo][tag][
            _idx]))  # & (blackhole_data['SubGroupNumber'] == sg_number[halo][tag][_idx])) TODO Satellites are commented out
        halo_mask, = np.where((subhalo_data['GroupNumber'] == gr_number[halo][tag][_idx]) & (
                subhalo_data['SubGroupNumber'] == sg_number[halo][tag][_idx]))

        # Mask the temporary dictionary for each galaxy #
        blackhole_data_tmp, subhalo_data_tmp = {}, {}
        for attribute in blackhole_data.keys():
            blackhole_data_tmp[attribute] = np.copy(blackhole_data[attribute])[blackhole_mask]
        for attribute in subhalo_data.keys():
            subhalo_data_tmp[attribute] = np.copy(subhalo_data[attribute])[halo_mask]

        # Normalise coordinates #
        for data in [blackhole_data_tmp]:
            data['Coordinates'] = data['Coordinates'] - subhalo_data_tmp['CentreOfPotential']

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
            axis.plot(bh_history['Age'], np.log10(bh_history['BH_Subgrid_Mass'] * 1e10), color=color)

        plots_path = '/cosma7/data/dp004/dc-irod1/FLARES/Lovell_21/plots/'  # Path to save plots.
        plt.savefig(plots_path + 'BH_positions' + '-' + date + '-' + str(halo) + '-' + str(_idx) + '.png',
                    bbox_inches='tight')
        plt.close()
print('Finished main.py in %.4s s' % (time.time() - start_time))
