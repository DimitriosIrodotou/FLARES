import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import eagle_IO.eagle_IO.eagle_IO as E

from flares import flares

start_time = time.time()
date = time.strftime('%d_%m_%y_%H%M')

# Load data #
fl = flares.flares(fname='/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5')
gr_number = fl.load_dataset('GroupNumber', arr_type='Galaxy')
sg_number = fl.load_dataset('SubGroupNumber', arr_type='Galaxy')
sfr = fl.load_dataset('SFR_aperture/30/inst', arr_type='Galaxy')
mstar = fl.load_dataset('Mstar_aperture/30', arr_type='Galaxy')

# Define the redshift #
z = 5.00
tag = '010_z005p000'
if z <= 2.8:
    softening = 0.000474390 / 0.6777 * 1e3  # In pkpc.
else:
    softening = 0.001802390 / (0.6777 * (1 + z)) * 1e3  # In pkpc.

# Create the plot
figure, axis = plt.subplots(1, figsize=(10, 7.5))

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
    centrals = (sg_number[halo][tag] == 0)
    satellites = (sg_number[halo][tag] > 0)
    idxs_centrals = np.where((ssfr < -1) & (_mstar > 9.69) & centrals)[0]
    idxs_satellites = np.where((ssfr < -1) & (_mstar > 9.69) & satellites)[0]
    if len(idxs_centrals) == 0 and len(idxs_satellites) == 0:
        continue
    else:
        print("Halo:", halo, "Galaxies:", len(_mstar), "Centrals/satellites:", len(np.where(centrals)[0]), "/",
              len(np.where(satellites)[0]), "Quenched centrals/satellites:", len(idxs_centrals), "/",
              len(idxs_satellites))

    # Extract black hole data #
    sim = "%s/GEAGLE_%s/data" % (fl.directory, halo)
    bh_mass = E.read_array('PARTDATA', sim, tag, '/PartType5/BH_Mass', numThreads=8, verbose=False)
    bh_pid = E.read_array('PARTDATA', sim, tag, '/PartType5/ParticleIDs', numThreads=8, verbose=False)
    bh_gr_number = E.read_array('PARTDATA', sim, tag, '/PartType5/GroupNumber', numThreads=8, verbose=False)
    bh_coordinates = E.read_array('PARTDATA', sim, tag, '/PartType5/Coordinates', numThreads=8, verbose=False)
    bh_sg_number = E.read_array('PARTDATA', sim, tag, '/PartType5/SubGroupNumber', numThreads=8, verbose=False)
    dat = pd.read_csv('/cosma7/data/dp004/dc-love2/codes/flares_passive/analysis/data/blackhole_details_h%s.csv' % halo)

    # Loop over all quenched and central galaxies in the particular halo #
    for _idx in idxs_centrals:
        print("ID:", _idx)

        # Select the most massive black hole in the group #
        # _mask = (bh_gr_number == gr_number[halo][tag][_idx])
        _mask, = np.where((bh_gr_number == gr_number[halo][tag][_idx]) & (bh_sg_number == sg_number[halo][tag][_idx]))

        # Use only the most massive black hole #
        PID = bh_pid[_mask]
        PID = PID[np.argmax(bh_mass[_mask])]

        mask = (dat['PID'] == PID)
        bh_history = dat.loc[mask].sort_values('Time')
        bh_history['z'] = (1. / bh_history['Time']) - 1
        bh_history['Age'] = fl.cosmo.age(bh_history['z'])

        plt.plot(bh_history['Age'], bh_history['BH_Subgrid_Mass'] * 1e4, color='C2')

plots_path = '/cosma7/data/dp004/dc-irod1/FLARES/Lovell_21/plots/'  # Path to save plots.
plt.savefig(plots_path + 'BH_accretion_rates' + '-' + date + '.png', bbox_inches='tight')
print('Finished main.py in %.4s s' % (time.time() - start_time))
