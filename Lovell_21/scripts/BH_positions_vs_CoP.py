import time
import plot_tools
import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
import eagle_IO.eagle_IO.eagle_IO as E

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
redshifts = [4.77, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00, 11.00, 12.00, 13.00, 14.00, 15.00]
tags = ['011_z004p770', '010_z005p000', '009_z006p000', '008_z007p000', '007_z008p000', '006_z009p000', '005_z010p000',
        '004_z011p000', '003_z012p000', '002_z013p000', '001_z014p000', '000_z015p000']

z, tag = 5.00, '010_z005p000'
# for z, tag in zip(redshifts, tags):
if z <= 2.8:
    softening = 0.000474390 / 0.6777 * 1e3  # In pkpc.
else:
    softening = 0.001802390 / (0.6777 * (1 + z)) * 1e3  # In pkpc.

# Create the plot

figure = plt.figure(figsize=(20, 20))

gs = gridspec.GridSpec(1, 2, wspace=0.3)
axis00, axis01 = figure.add_subplot(gs[0, 0]), figure.add_subplot(gs[0, 1])
plot_tools.set_axes(axis00, xlim=[1e5, 1e10], ylim=[1e-5, 1e5], yscale='log', xscale='log',
                    xlabel=r'$\mathrm{M_\bullet/M_\odot}$', ylabel=r'$\mathrm{(r_\bullet - r_{CoP})/kpc}$', size=30,
                    which='major')

plot_tools.set_axes(axis01, [1e5, 1e10], ylim=[1e-3, 1e3], yscale='log', xscale='log',
                    xlabel=r'$\mathrm{M_\bullet/M_\odot}$', ylabel=r'$\mathrm{|r_{CoM} - r_{CoP}|/kpc}$', size=30,
                    which='major')

# Loop over the 40 haloes #
# fl.halos = ['03']
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
    if len(idxs_centrals) == 0:  # and len(idxs_satellites) == 0: TODO Satellites are commented out
        continue
    else:
        print("Halo:", halo, "Galaxies:", len(_mstar), "Centrals/satellites:", len(np.where(centrals)[0]), "/",
              len(np.where(satellites)[0]), "Quenched centrals/satellites:", len(idxs_centrals), "/",
              len(idxs_satellites))

    # Extract stellar, black hole and halo data #
    sim = "%s/GEAGLE_%s/data" % (fl.directory, halo)
    blackhole_data, subhalo_data = {}, {}
    for attribute in ['BH_Mass', 'Coordinates', 'GroupNumber', 'SubGroupNumber']:
        blackhole_data[attribute] = E.read_array('PARTDATA', sim, tag, '/PartType5/' + attribute, numThreads=8)
    for attribute in ['CentreOfMass', 'CentreOfPotential', 'GroupNumber', 'SubGroupNumber']:
        subhalo_data[attribute] = E.read_array('SUBFIND', sim, tag, '/Subhalo/' + attribute, numThreads=8)
    blackhole_details = pd.read_csv(
        '/cosma7/data/dp004/dc-love2/codes/flares_passive/analysis/data/blackhole_details_h%s.csv' % halo)

    blackhole_data['BH_Mass'] *= u.g.to(u.Msun)
    subhalo_data['CentreOfMass'] *= u.cm.to(u.kpc)
    blackhole_data['Coordinates'] *= u.cm.to(u.kpc)
    subhalo_data['CentreOfPotential'] *= u.cm.to(u.kpc)

    # Loop over all quenched and central galaxies in the particular halo #
    # idxs_centrals = [104]
    for _idx in idxs_centrals:

        # For a specific halo, redshift and "ID" print the BH_Mass and SubGroupNumber belonging in this GroupNumber #
        gn_mask, = np.where(blackhole_data['GroupNumber'] == gr_number[halo][tag][_idx])
        # print(gr_number[halo][tag][_idx])
        # print(blackhole_data['GroupNumber'][gn_mask])
        # print(blackhole_data['BH_Mass'][gn_mask])
        # print(blackhole_data['SubGroupNumber'][gn_mask])

        # Select the most massive BH in the Group and get its SubGroupNumber #
        smbh_mask, = np.where((blackhole_data['BH_Mass'][gn_mask] == max(blackhole_data['BH_Mass'][gn_mask])) & (
                blackhole_data['SubGroupNumber'][gn_mask] > 0))
        # print(blackhole_data['SubGroupNumber'][gn_mask][smbh_mask])
        # print(blackhole_data['BH_Mass'][gn_mask][smbh_mask])

        # Get masks for each specific galaxy #
        halo_mask, = np.where((subhalo_data['GroupNumber'] == gr_number[halo][tag][_idx]) & (
                subhalo_data['SubGroupNumber'] == sg_number[halo][tag][_idx]))

        # Mask the temporary dictionary for each galaxy #
        subhalo_data_tmp = {}
        for attribute in subhalo_data.keys():
            subhalo_data_tmp[attribute] = np.copy(subhalo_data[attribute])[halo_mask]

        # Normalise coordinates #
        if len(blackhole_data['BH_Mass'][gn_mask][smbh_mask]) == 0:
            blackhole_mask = np.where((blackhole_data['GroupNumber'] == gr_number[halo][tag][
                _idx]) & (blackhole_data['SubGroupNumber'] == sg_number[halo][tag][_idx]))
            distances = blackhole_data['Coordinates'][blackhole_mask] - subhalo_data_tmp['CentreOfPotential']

            # Many black holes belong in the same structure with one CoP and CoM
            distance_diffence = np.ones(len(blackhole_data['Coordinates'][blackhole_mask]), dtype='f8') * (
                    np.linalg.norm(subhalo_data_tmp['CentreOfMass']) - np.linalg.norm(
                subhalo_data_tmp['CentreOfPotential']))

            axis00.scatter(blackhole_data['BH_Mass'][blackhole_mask], np.linalg.norm(distances, axis=1), color='k')
            axis01.scatter(blackhole_data['BH_Mass'][blackhole_mask], np.abs(distance_diffence), color='k')
        else:
            distances = blackhole_data['Coordinates'][gn_mask][smbh_mask] - subhalo_data_tmp['CentreOfPotential']

            axis00.scatter(blackhole_data['BH_Mass'][gn_mask][smbh_mask], np.linalg.norm(distances, axis=1),
                           color='tab:red')
            axis01.scatter(blackhole_data['BH_Mass'][gn_mask][smbh_mask],
                           np.abs(np.linalg.norm(subhalo_data_tmp['CentreOfMass']) - np.linalg.norm(
                               subhalo_data_tmp['CentreOfPotential'])), color='tab:red')

axis00.text(0.02, 0.85,
            r'$z = %.2f \;$' r'$\mathrm{Centrals}$'  '\n'  r'$\mathrm{M_\bigstar/M_\odot >10^{9.7}} \;$'
            r'$\mathrm{sSFR \times Gyr < -1 }$' % z, transform=axis00.transAxes, size=20)
plots_path = '/cosma7/data/dp004/dc-irod1/FLARES/Lovell_21/plots/'  # Path to save plots.
plt.savefig(plots_path + 'BH_positions_vs_CoP' + '-' + date + '-' + str(z) + '.png', bbox_inches='tight')
plt.close()
print('Finished main.py in %.4s s' % (time.time() - start_time))
