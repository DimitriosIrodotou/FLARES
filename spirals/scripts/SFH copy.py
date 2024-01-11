import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

import astropy.units as u
from astropy.cosmology import z_at_value

import eagle_IO.eagle_IO.eagle_IO as E

from flares import flares

date = time.strftime('%d_%m_%y_%H%M')  # Date.
fl = flares.flares(fname='/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5')


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


mstar = fl.load_dataset('Mstar_aperture/30', arr_type='Galaxy')
sfr = fl.load_dataset('SFR_aperture/30/inst', arr_type='Galaxy')
# sfr = fl.load_dataset('SFR/SFR_10',arr_type='Galaxy')
bhmass = fl.load_dataset('BH_Mass', arr_type='Galaxy')
sgrp = fl.load_dataset('SubGroupNumber', arr_type='Galaxy')
grp = fl.load_dataset('GroupNumber', arr_type='Galaxy')

tag = '010_z005p000'
z = 5.00

## multi-plot set up
fig, axes = plt.subplots(4, 3, figsize=(20, 26))
plt.subplots_adjust(wspace=0.8)
axes = axes.flatten()
axidx = 0

# halo = '08'
for halo in fl.halos:

    _p = fl.get_particles(p_str=['S_Age', 'S_Mass', 'S_Z', 'S_MassInitial'],
                          length_str='S_Length', halo=halo, tag=tag, verbose=False)

    _mstar = mstar[halo][tag] * 1e10
    _sfr = sfr[halo][tag]
    _bhmass = bhmass[halo][tag] * 1e10

    _sfr[_sfr <= 0] = 1E-10
    _mstar[_mstar <= 0] = 1E4
    _sfr = np.log10(_sfr)
    _mstar = np.log10(_mstar)

    ssfr = _sfr - _mstar
    ssfr += 9  # convert to Gyr**-1

    centrals = (sgrp[halo][tag] == 0)

    idxs = np.where((ssfr < -1) & (_mstar > 9.69) & centrals)[0]
    print("selected halo/idxs:", halo, idxs)

    if (axidx + len(idxs)) > 11:  # filter idxs to fit plot grid
        if (12 - axidx) > 0:
            idxs = [idxs[11 - axidx]]
        else:
            idxs = []

    print("plotted halo/idxs:", halo, idxs, axidx)

    if len(idxs) > 0:

        ## find PID of central black hole
        sim = "%s/GEAGLE_%s/data" % (fl.directory, halo)
        bh_grp = E.read_array('PARTDATA', sim, tag, '/PartType5/GroupNumber',
                              numThreads=8, verbose=False)
        bh_sgrp = E.read_array('PARTDATA', sim, tag, '/PartType5/SubGroupNumber',
                               numThreads=8, verbose=False)
        bh_pid = E.read_array('PARTDATA', sim, tag, '/PartType5/ParticleIDs',
                              numThreads=8, verbose=False)
        bh_mass = E.read_array('PARTDATA', sim, tag, '/PartType5/BH_Mass',
                               numThreads=8, verbose=False)
        bh_last_accr = E.read_array('PARTDATA', sim, tag, '/PartType5/BH_TimeLastMerger',
                                    numThreads=8, verbose=False)

        dat = pd.read_csv(
            '/cosma7/data/dp004/dc-love2/codes/flares_passive/analysis/data/blackhole_details_h%s.csv' % halo)

        for _idx in idxs:
            # _idx = idxs[0]

            p_age = _p[_idx]['S_Age']
            p_imass = _p[_idx]['S_MassInitial'] * 1e10
            p_Z = np.log10(_p[_idx]['S_Z'] / 0.02)  # solar metallicity

            universe_age = fl.cosmo.age(z).value
            form_time = universe_age - p_age

            ## ---- black hole history

            ## sometimes the most massive black hole is not in the subhalo, but in the group. 
            ## Not sure why, perhaps interactions?
            _mask = (bh_grp == grp[halo][tag][_idx]) & (bh_sgrp == sgrp[halo][tag][_idx])
            _mask = (bh_grp == grp[halo][tag][_idx])  # & (bh_sgrp == sgrp[halo][tag][_idx])

            print("whole halo |", _idx, np.log10(_bhmass[_idx]), np.log10(np.sum(bh_mass[_mask]) * 1e10))
            print(np.log10(bh_mass[_mask] * 1e10))
            print("#######\n\n")
            print("Any mergers? [Non-zero]", bh_last_accr[_mask])

            PID = bh_pid[_mask]
            PID = PID[np.argmax(bh_mass[_mask])]
            last_accretion = bh_last_accr[np.argmax(bh_mass[_mask])]
            if last_accretion > 0.0: last_accretion = fl.cosmo.age((1. / last_accretion) - 1).value

            mask = (dat['PID'] == PID)

            bh_history = dat.loc[mask].sort_values('Time')

            bh_history['z'] = (1. / bh_history['Time']) - 1
            bh_history['Age'] = fl.cosmo.age(bh_history['z'])

            last_output = bh_history['Age'].iloc[-1]
            first_output = bh_history['Age'].iloc[0]
            details_mass = bh_history['BH_Subgrid_Mass'].iloc[-1] * 1e10

            binLimits = np.linspace(0, universe_age, 15)
            bins = binLimits[:-1] + (np.diff(binLimits) / 2)
            ZbinLimits = np.linspace(-4, 1.0, 30)

            ax1 = axes[axidx]
            axidx += 1

            # fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,figsize=(5,18))
            # fig, (ax1,ax2) = plt.subplots(2,1,figsize=(5,10))
            # fig, ax1 = plt.subplots(1,1)#,figsize=(5,10))
            # plt.subplots_adjust(hspace=0.)

            # ax2.hist2d(form_time,p_Z,cmin=1,bins=(binLimits,ZbinLimits))
            # ax2.set_ylabel('$Z \,/\, Z_{\odot}$', size=13)
            # ax2.set_ylim(-4,0.9)
            # ax2.grid(alpha=0.5,axis='x')

            counts, dummy = np.histogram(form_time, bins=binLimits, weights=p_imass)  # Msol
            counts /= np.diff(binLimits) * 1e9  # Msol / yr
            plot_bins = np.hstack([binLimits[1:], bins[-1] + np.diff(bins)[0]])
            ax1.step(plot_bins, np.hstack([counts, counts[-1]]), where='pre', color='C0')

            ax3 = ax1.twinx()
            ax3.spines["right"].set_position(("axes", 1.3))
            make_patch_spines_invisible(ax3)  # activate frame, but make patch+spines invisible.
            ax3.spines["right"].set_visible(True)  # Second, show the right spine.

            ax3.plot(bh_history['Age'], bh_history['BH_Subgrid_Mass'] * 1e4, color='C2')
            ax3.set_ylabel('$M_{\\bullet} \,/\, (10^{6} \, \mathrm{M_{\odot}})$', size=13)
            if (_bhmass[_idx] / details_mass) > 10:  ax3.text(0.1, 0.9, 'Black hole history incomplete!',
                                                              transform=ax3.transAxes, color='red', size=16)

            ax4 = ax1.twinx()
            _n, _dummy, _dummy = ax4.hist(bh_history['Age'], bins=binLimits, weights=bh_history['Mdot'], color='C1',
                                          alpha=0.5)
            if first_output > 0.: ax4.arrow(first_output, 0.1, 0., -0.09, color='green', transform=ax4.transAxes,
                                            width=0.01)
            ax4.arrow(last_output, 0.1, 0., -0.09, color='red', transform=ax4.transAxes, width=0.01)
            if last_accretion > 0.: ax4.arrow(last_accretion, 0.1, 0., -0.09, color='orange', transform=ax4.transAxes,
                                              width=0.01)
            ax4.set_ylabel('$\dot{M}_{\\bullet} \,/\, ( \mathrm{M_{\odot} \; yr^{-1}} )$', size=13)

            ax1.set_xlabel('$\mathrm{Age \,/\, Gyr}$', size=13)
            ax1.set_ylabel('$\mathrm{SFR} \,/\, (M_{\odot} \; \mathrm{yr^{-1}})$', size=13)
            ax1.text(0.05, 0.82, '$\mathrm{log_{10}}(M_{*} \,/\, M_{\odot}) = %.2f$' % _mstar[_idx],
                     transform=ax1.transAxes, color='grey')
            ax1.text(0.05, 0.74, '$\mathrm{SFR}(z = 5) = %.2f$' % (10 ** _sfr[_idx]), transform=ax1.transAxes,
                     color='grey')
            ax1.text(0.05, 0.66, '$\mathrm{log_{10}}(M_{BH} \,/\, M_{\odot}) = %.2f$' % (np.log10(_bhmass[_idx])),
                     transform=ax1.transAxes, color='grey')
            ax1.text(0.05, 0.9, '$\mathrm{Region: \; %s \; | \; Galaxy: \; %s}$' % (halo, _idx),
                     transform=ax1.transAxes, color='grey')

            for ax in [ax1, ax3, ax4]: ax.set_xlim(0, universe_age)
            # ax1.set_xticklabels([])
            for ax in [ax1, ax3, ax4]: ax.set_ylim(0, )

            ax1t = ax1.twiny()
            _ticks = fl.cosmo.age([5, 6, 7, 8, 9, 10, 12, 15, 20, 30]).value
            ax1t.set_xticks(_ticks);
            ax1t.set_xticklabels(["%.0f" % z_at_value(fl.cosmo.age, _b * u.Gyr) for _b in _ticks])
            ax1t.set_xlabel('$z$', size=13)
            ax1.yaxis.label.set_color('C0')
            ax3.yaxis.label.set_color('C2')
            ax4.yaxis.label.set_color('C1')
            # plt.show()
            # plt.savefig(f'images/SFH_h%s_%s.png'%(halo,_idx),dpi=200,bbox_inches='tight')

plots_path = '/cosma7/data/dp004/dc-irod1/FLARES/Lovell_21/plots/'  # Path to save plots.
plt.savefig(plots_path + 'SFH' + '-' + date + '.png', bbox_inches='tight')
# plt.show()
# plt.savefig(f'images/SFH_multi.png', dpi=200, bbox_inches='tight')
