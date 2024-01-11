import time
import numpy as np

from scipy import stats

import matplotlib.pyplot as plt

from flares import flares

date = time.strftime('%d_%m_%y_%H%M')  # Date.
fl = flares.flares(fname='/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5')

mstar = fl.load_dataset('Mstar_aperture/30', arr_type='Galaxy')
sfr = fl.load_dataset('SFR_aperture/30/100Myr', arr_type='Galaxy')

xlims = np.array([9, 11.5])
ylims = np.array([-2.5, 2.])

out = {tag: None for tag in fl.tags}
# --- all simulations (un-weighted)

for tag, z in zip([fl.tags[5]], [fl.zeds[5]]):
    _mstar = np.hstack([mstar[halo][tag] for halo in fl.halos]) * 1e10
    _sfr = np.hstack([sfr[halo][tag] for halo in fl.halos])

    _sfr[_sfr <= 0] = 1E-10
    _mstar[_mstar <= 0] = 1E4
    _sfr = np.log10(_sfr)
    _mstar = np.log10(_mstar)

    ssfr = _sfr - _mstar
    ssfr += 9  # convert to Gyr**-1

    # sSFR #
    s = (ssfr < -1)  # &(_mstar>xlims[0])
    print("z:", z, "| N(Q):", len(ssfr[s]))

    # dMS #
    # --- fit by a linear slope
    slope, intercept, r_value, p_value, std_err = stats.linregress(_mstar[_mstar > 9.5], ssfr[_mstar > 9.5])
    R = ssfr[_mstar > 9.5] - (slope * _mstar[_mstar > 9.5] + intercept)
    R16 = np.percentile(R, 16.0)
    R84 = np.percentile(R, 84.0)
    Rstd = np.std(R)
    Rsig = (R84 - R16) / 2
    # print(R16,R84,Rstd,Rsig)

    R = ssfr - (slope * _mstar + intercept)
    sel = R < -Rsig * 7

    # save selection
    out[tag] = list(np.where(s & sel)[0])

    # Start figure
    fig, ax = plt.subplots(1, 1)

    ax.scatter(_mstar, ssfr, s=3, alpha=1, c='0.9', zorder=1, lw=0)
    ax.hexbin(_mstar, ssfr, gridsize=(30, 14), extent=(*xlims, -1, ylims[1]), bins='log', cmap='Greys', linewidths=0.,
              mincnt=2, alpha=1.0, zorder=2)  # , vmin = 5, vmax = 100)

    # --- sSFR cut
    ax.fill_between([9., 12], [-1.0, -1.0], [-3, -3], color='k', alpha=0.05)
    ax.fill_between([9., 12], [-2.0, -2.0], [-3, -3], color='k', alpha=0.05)
    ax.plot([9., 12], [-1.0, -1.0], c='k', lw=1)
    ax.plot([9., 12], [-2.0, -2.0], c='k', lw=1, linestyle='dashed')

    # --- dMS
    _x = np.array([9.5, xlims[1]])
    ax.plot(_x, slope * _x + intercept, color='C0')
    _x = np.array([xlims[0], 9.5])
    ax.plot(_x, slope * _x + intercept, linestyle='dashed', color='C0')

    ax.fill_between(xlims, slope * xlims + intercept - Rsig, slope * xlims + intercept + Rsig, alpha=0.2, zorder=3)

    ax.plot(xlims, slope * xlims + intercept - Rsig * 7, c='C0', linestyle='dotted')
    ax.fill_between(xlims, slope * xlims + intercept - Rsig * 7, [-3, -3], color='k', alpha=0.05)

    ax.scatter(_mstar[s | sel], ssfr[s | sel], s=5, alpha=1, c='k', zorder=3, lw=0)

    # bins = np.arange(*xlims, 0.2)
    # P16, P50, P84 = fplt.average_line(mstar, ssfr,bins)
    # ax.fill_between(bins,P84,P16,color='k', alpha=0.15)
    # ax.plot(bins,P50,ls='-',c='k', alpha=1.0, lw=1)

    ax.text(0.8, 0.9, '$z={z}$', transform=ax.transAxes, size=13)

    ax.set_ylim(ylims)
    ax.set_xlim(xlims)

    ax.set_xticks(np.arange(xlims[0], xlims[1] + 0.1, 0.5))

    ax.set_ylabel(r'$\log_{10}({\rm sSFR}/{\rm Gyr^{-1})}$', size=13)
    ax.set_xlabel(r'$\log_{10}(M_{\star}/{\rm M_{\odot}})$', size=13)
    ax.grid(True)

    plots_path = '/cosma7/data/dp004/dc-irod1/FLARES/Lovell_21/plots/'  # Path to save plots.
    plt.savefig(plots_path + 'sSFR' + '-' + date + '.png', bbox_inches='tight')
