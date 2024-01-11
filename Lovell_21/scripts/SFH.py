import h5py
import numpy as np
import pandas as pd

from scipy.stats import binned_statistic

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

import astropy.units as u
from astropy.cosmology import z_at_value   

import eagle_IO.eagle_IO as E

import flares
fl = flares.flares(fname='../../flares/data/flares.hdf5')


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

bhmass = fl.load_dataset('BH_Mass',arr_type='Galaxy')
sgrp = fl.load_dataset('SubGroupNumber',arr_type='Galaxy')
grp = fl.load_dataset('GroupNumber',arr_type='Galaxy')
cop = fl.load_dataset('COP', arr_type='Galaxy')

tag = '010_z005p000'
z = 5.00
p_sel_str = 'sSFR < -1'

## multi-plot set up
fig, axes = plt.subplots(3,2,figsize=(12,16))
plt.subplots_adjust(wspace=0.8, hspace=0.3)
axes = axes.flatten()
axidx = 0

for region in fl.halos:

    timescale = '50Myr'
    with h5py.File('data/select_quiescent.h5','r') as hf:
        _mstar = hf[f'{tag}/{region}/Mstar'][:]
        _ssfr  = hf[f'{tag}/{region}/sSFR/{timescale}'][:]
        _sfr  = hf[f'{tag}/{region}/SFR/{timescale}'][:]
        _quies = hf[f'{tag}/{region}/quiescent/{p_sel_str}/{timescale}'][:]

    _p = fl.get_particles(p_str = ['S_Age', 'S_Mass', 'S_Z', 'S_MassInitial', 'S_Coordinates'], 
                          length_str='S_Length', halo=region, tag=tag, verbose=False) 
    
    _bhmass = bhmass[region][tag] * 1e10
    centrals = (sgrp[region][tag] == 0)
    idxs = np.where(_quies & (_ssfr < -1) & (_mstar > 9.7) & centrals)[0]
    print("selected region/idxs:", region, idxs)

    if (len(idxs) > 0) & (axidx < 12):

        ## find PID of central black hole
        sim = "%s/GEAGLE_%s/data"%(fl.directory,region)
        bh_grp =  E.read_array('PARTDATA', sim,tag, '/PartType5/GroupNumber', 
                   numThreads=8, verbose=False)
        bh_sgrp = E.read_array('PARTDATA', sim,tag, '/PartType5/SubGroupNumber', 
                   numThreads=8, verbose=False)
        bh_pid =  E.read_array('PARTDATA', sim,tag, '/PartType5/ParticleIDs', 
                   numThreads=8, verbose=False)
        bh_mass =  E.read_array('PARTDATA', sim,tag, '/PartType5/BH_Mass', 
                   numThreads=8, verbose=False)
        bh_last_accr =  E.read_array('PARTDATA', sim,tag, '/PartType5/BH_TimeLastMerger', 
                   numThreads=8, verbose=False)
        # bh_prog_id =  E.read_array('PARTDATA', sim,tag, '/PartType5/BH_MostMassiveProgenitorID',
        #            numThreads=8, verbose=False)
    
        dat = pd.read_csv('data/blackhole_details_h%s.csv'%region, float_precision='round_trip')

        with h5py.File('data/select_quiescent.h5','r') as hf:
            _aperture = hf[f'{tag}/{region}/Aperture'][:]

        ## ignore 
        if region == '03':
            idxs = np.delete(idxs, 2)

        for _idx in idxs:
            if axidx == 6:
                plt.savefig(f'images/SFH_multi.png', dpi=200, bbox_inches='tight')
                plt.close()
                fig, axes = plt.subplots(3,2,figsize=(12,16))
                plt.subplots_adjust(wspace=0.8, hspace=0.3)
                axes = axes.flatten()
            elif axidx > 11:
                break
 
            
            if axidx > 5:
                ax1 = axes[axidx-6]
            else:
                ax1 = axes[axidx]
            print(axidx)
            axidx+=1           
            print("plotting", _idx)
            
            ## ---- create aperture mask
            ap_mask = np.sqrt(np.sum((_p[_idx]['S_Coordinates'].T - \
                            cop[region][tag][:,_idx])**2, axis=1)) < _aperture[_idx]


            p_age = _p[_idx]['S_Age'][ap_mask]
            p_imass = _p[_idx]['S_MassInitial'][ap_mask] * 1e10
            # p_Z = np.log10(_p[_idx]['S_Z'][ap_mask] / 0.02) # solar metallicity
            
            universe_age = fl.cosmo.age(z).value
            form_time = universe_age - p_age
            
            ## ---- black hole history

            ## sometimes the most massive black hole is not in the subhalo, but in the group. 
            ## Not sure why, perhaps interactions?
            # _mask = (bh_grp == grp[region][tag][_idx]) & (bh_sgrp == sgrp[region][tag][_idx])
            _mask = (bh_grp == grp[region][tag][_idx])# & (bh_sgrp == sgrp[region][tag][_idx])

            # print("whole halo |",_idx, np.log10(_bhmass[_idx]), np.log10(np.sum(bh_mass[_mask]) * 1e10))
            # print(np.log10(bh_mass[_mask] * 1e10))
            # print("#######\n\n")
            # print("Any mergers? [Non-zero]",bh_last_accr[_mask])
            
            PID = bh_pid[_mask]
            PID = PID[np.argmax(bh_mass[_mask])]
            mask = (dat['PID'] == PID)
            bh_history = dat.loc[mask].sort_values('Time')
            bh_history['z'] = (1. / bh_history['Time']) - 1
            bh_history['Age'] = fl.cosmo.age(bh_history['z'])
            # last_output = bh_history['Age'].iloc[-1]
            first_output = bh_history['Age'].iloc[0]
            details_mass = bh_history['BH_Subgrid_Mass'].iloc[-1] * 1e10
            
            # if bh_prog_id[_mask][np.argmax(bh_mass[_mask])] != PID:
            #     print("we have a progenitor!", bh_prog_id[_mask][np.argmax(bh_mass[_mask])], PID)

            binLimits = np.linspace(0,universe_age,20)
            bins = binLimits[:-1] + (np.diff(binLimits) / 2)
            # ZbinLimits = np.linspace(-4,1.0,30)
            

            #ax2.hist2d(form_time,p_Z,cmin=1,bins=(binLimits,ZbinLimits))
            #ax2.set_ylabel('$Z \,/\, Z_{\odot}$', size=13)
            #ax2.set_ylim(-4,0.9)
            #ax2.grid(alpha=0.5,axis='x')
            
            counts,dummy = np.histogram(form_time,bins=binLimits,weights=p_imass) # Msol
            counts /= np.diff(binLimits)*1e9 # Msol / yr
            plot_bins = np.hstack([binLimits[1:],bins[-1]+np.diff(bins)[0]])
            ax1.step(plot_bins, np.hstack([counts,counts[-1]]), where='pre', color='C0')
            
            ax3 = ax1.twinx()
            ax3.spines["right"].set_position(("axes", 1.3)) 
            make_patch_spines_invisible(ax3) # activate frame, but make patch+spines invisible.
            ax3.spines["right"].set_visible(True) # Second, show the right spine.

            ax3.plot(bh_history['Age'], bh_history['BH_Subgrid_Mass'] * 1e4, color='C2')
            ax3.set_ylabel('$M_{\\bullet} \,/\, (10^{6} \, \mathrm{M_{\odot}})$', size=13)
            if (_bhmass[_idx] / details_mass) > 10:  ax3.text(0.1,0.9,'Black hole history incomplete!',
                         transform=ax3.transAxes, color='red', size=16)

            ax4 = ax1.twinx()

            _sums, _dummy, _dummy = binned_statistic(bh_history['Age'], 
                                                     bh_history['BH_Subgrid_Mass'] * 1e10, 
                                                     statistic= lambda x: np.max(x) - np.min(x), 
                                                     bins=binLimits)
            _mdot = _sums / np.diff(binLimits * 1e9)
            ax4.bar(bins, _mdot, width=np.diff(binLimits), color='C1', alpha=0.5)
            
            # ax4.plot(bh_history['Age'], bh_history['Mdot'], color='C1')
            # _n, _dummy, _dummy = ax4.hist(bh_history['Age'], bins=binLimits, \
            #         weights=bh_history['Mdot'], color='C1', alpha=0.5)
            if first_output > 0.: ax4.arrow(first_output / universe_age, 0.1, 0., -0.09, color='green', 
                                            width=0.01, transform=ax4.transAxes)
            # ax4.arrow(last_output, 0.1, 0., -0.09, color='red', transform=ax4.transAxes, width=0.01)
            # if last_accretion > 0.: ax4.arrow(last_accretion, 0.1, 0.,-0.09,color='orange', transform=ax4.transAxes, width=0.01)
            ax4.set_ylabel('$\dot{M}_{\\bullet} \,/\, ( \mathrm{M_{\odot} \; yr^{-1}} )$', size=13)
            
            # last_accretion = bh_last_accr[_mask]#[np.argmax(bh_mass[_mask])]
            # last_accretion = 1. / last_accretion[last_accretion > 0] - 1
            # # if last_accretion > 0.0: last_accretion = fl.cosmo.age((1./ last_accretion) - 1).value
            # for _last in last_accretion:
            #     ax4.arrow(fl.cosmo.age(_last).value / universe_age, 0.1, 0., -0.09, color='orange', 
            #               transform=ax4.transAxes, width=0.01)

            ax1.set_xlabel('$\mathrm{Age \,/\, Gyr}$', size=13)            
            ax1.set_ylabel('$\mathrm{SFR} \,/\, (M_{\odot} \; \mathrm{yr^{-1}})$',size=13)
            # ax1.text(0.05, 0.9, '$\mathrm{Region: \; %s \; | \; Galaxy: \; %s}$'%(region,_idx), 
            #          transform=ax1.transAxes, color='grey')
            ax1.text(0.05,0.94,'$\mathrm{log_{10}}(M_{*} \,/\, M_{\odot}) = %.2f$'%_mstar[_idx],
                     transform=ax1.transAxes,color='grey')
            ax1.text(0.05,0.86,'$\mathrm{log_{10}(sSFR \,/\, Gyr)} = %.2f$'%_ssfr[_idx], transform=ax1.transAxes,color='grey')
            ax1.text(0.05,0.78,'$\mathrm{log_{10}}(M_{BH} \,/\, M_{\odot}) = %.2f$'%(np.log10(_bhmass[_idx])),
                     transform=ax1.transAxes,color='grey')
            # ax1.text(0.05,0.7,'$\mathrm{Central?} = %i$'%(sgrp[region][tag][_idx] == 0),
            #          transform=ax1.transAxes,color='grey')

            for ax in [ax1,ax3,ax4]: ax.set_xlim(0,universe_age)
            # ax1.set_xticklabels([])
            for ax in [ax1,ax3,ax4]: ax.set_ylim(0,)

            ax1t = ax1.twiny()
            _ticks = fl.cosmo.age([5,6,7,8,9,10,12,15,20,30]).value
            ax1t.set_xticks(_ticks);
            ax1t.set_xticklabels(["%.0f"%z_at_value(fl.cosmo.age,_b * u.Gyr) for _b in _ticks])
            ax1t.set_xlabel('$z$',size=13)
            ax1.yaxis.label.set_color('C0')
            ax3.yaxis.label.set_color('C2')
            ax4.yaxis.label.set_color('C1') 
            # plt.show()
            # plt.savefig(f'images/SFH_h%s_%s.png'%(halo,_idx),dpi=200,bbox_inches='tight')

# plt.show()
# plt.savefig(f'images/SFH_multi.png', dpi=200, bbox_inches='tight') 
plt.savefig(f'images/SFH_multi_B.png', dpi=200, bbox_inches='tight') 
