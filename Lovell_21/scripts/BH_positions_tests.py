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
    gas_data, stellar_data, blackhole_data, dark_matter_data = {}, {}, {}, {}
    for attribute in ['Mass', 'GroupNumber', 'SubGroupNumber']:
        gas_data[attribute] = E.read_array('PARTDATA', sim, tag, '/PartType0/' + attribute, numThreads=8)
    for attribute in ['GroupNumber', 'SubGroupNumber']:
        dark_matter_data[attribute] = E.read_array('PARTDATA', sim, tag, '/PartType1/' + attribute, numThreads=8)
    for attribute in ['Mass', 'GroupNumber', 'SubGroupNumber']:
        stellar_data[attribute] = E.read_array('PARTDATA', sim, tag, '/PartType4/' + attribute, numThreads=8)
    for attribute in ['BH_Mass', 'GroupNumber', 'SubGroupNumber']:
        blackhole_data[attribute] = E.read_array('PARTDATA', sim, tag, '/PartType5/' + attribute, numThreads=8)
    blackhole_details = pd.read_csv(
        '/cosma7/data/dp004/dc-love2/codes/flares_passive/analysis/data/blackhole_details_h%s.csv' % halo)

    gas_data['Mass'] *= u.g.to(u.Msun)
    stellar_data['Mass'] *= u.g.to(u.Msun)
    blackhole_data['BH_Mass'] *= u.g.to(u.Msun)

    # Loop over all quenched and central galaxies in the particular halo #
    centrals = [104]
    for _idx in centrals:
        # for _idx in idxs_centrals:

        # For a specific halo, redshift and "ID" print the BH_Mass and SubGroupNumber belonging in this GroupNumber #
        gn_mask, = np.where(blackhole_data['GroupNumber'] == gr_number[halo][tag][_idx])
        print(gr_number[halo][tag][_idx])
        print(blackhole_data['GroupNumber'][gn_mask])
        print(blackhole_data['BH_Mass'][gn_mask])
        print(blackhole_data['SubGroupNumber'][gn_mask])

        # Select the most massive BH in the Group and get its SubGroupNumber #
        smbh_mask, = np.where((blackhole_data['BH_Mass'][gn_mask] == max(blackhole_data['BH_Mass'][gn_mask])) & (
                blackhole_data['SubGroupNumber'][gn_mask] > 0))
        smbh_sgn_mask, = np.where(
            stellar_data['SubGroupNumber'] == blackhole_data['SubGroupNumber'][gn_mask][smbh_mask])
        print(blackhole_data['SubGroupNumber'][gn_mask][smbh_mask])
        print(blackhole_data['BH_Mass'][gn_mask][smbh_mask])

        # Select stellar particles associated with the most massive BH based on their common SubGroupNumber #
        print(stellar_data['SubGroupNumber'][smbh_sgn_mask])
        print(len(stellar_data['Mass'][smbh_sgn_mask]))
        print(np.log10(np.sum(stellar_data['Mass'][smbh_sgn_mask])))

        # Select gas particles associated with the most massive BH based on their common SubGroupNumber #
        gas_smbh_sgn_mask, = np.where(
            gas_data['SubGroupNumber'] == blackhole_data['SubGroupNumber'][gn_mask][smbh_mask])
        print(gas_data['SubGroupNumber'][gas_smbh_sgn_mask])
        print(len(gas_data['Mass'][gas_smbh_sgn_mask]))
        print(np.log10(np.sum(gas_data['Mass'][gas_smbh_sgn_mask])))

        # Select dark matter particles associated with the most massive BH based on their common SubGroupNumber #
        dm_smbh_sgn_mask, = np.where(
            dark_matter_data['SubGroupNumber'] == blackhole_data['SubGroupNumber'][gn_mask][smbh_mask])
        print(dark_matter_data['SubGroupNumber'][dm_smbh_sgn_mask])
        print(len(dark_matter_data['SubGroupNumber'][dm_smbh_sgn_mask]))

print('Finished main.py in %.4s s' % (time.time() - start_time))
