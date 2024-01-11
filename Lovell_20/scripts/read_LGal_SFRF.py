import time
import numpy as np
import matplotlib.pyplot as plt

date = time.strftime("%d\%m\%y\%H%M")

# Read HWT15 data for redshift 5 to 10 #
snapnums = [11, 12, 13, 15, 17, 19]
for snapnum in snapnums:
    HWT15 = np.loadtxt('snap_' + str(snapnum) + '_2015.txt',
                       dtype={'names': ('type', 'stellar_mass', 'sfr'), 'formats': (np.int, np.float, np.float)}, delimiter=',', skiprows=12)

    # Read HYF20 data for redshift 5 to 10 #
    HWT20 = np.loadtxt('snap_' + str(snapnum) + '_2020.txt',
                       dtype={'names': ('type', 'stellar_mass', 'sfr'), 'formats': (np.int, np.float, np.float)}, delimiter=',', skiprows=12)

    # Put into bins and normalise to number per unit volume (Mpc/h) per dex
    hubble = 0.673  # [dimensionless]
    boxside = 480.28 / hubble  # [Mpc]

    firstFile = 0
    lastFile = 511
    maxFile = 512
    binperdex = 5
    xrange = np.array([0.0, 3.0])
    nbin = (xrange[1] - xrange[0]) * binperdex

    nobj15, bins15, junk15 = plt.hist(np.log10(HWT15['sfr']), bins=np.int(nbin), range=xrange, log=True)
    yHen15 = nobj15 * maxFile / ((lastFile - firstFile + 1) * boxside ** 3) * binperdex
    xHen15 = 0.5 * (bins15[:-1] + bins15[1:])
    data = np.vstack((xHen15, yHen15))
    np.save('data_' + str(snapnum) + '_2015_SFRF', data)

    nobj20, bins20, junk20 = plt.hist(np.log10(HWT20['sfr']), bins=np.int(nbin), range=xrange, log=True)
    yHen20 = nobj20 * maxFile / ((lastFile - firstFile + 1) * boxside ** 3) * binperdex
    xHen20 = 0.5 * (bins20[:-1] + bins20[1:])
    data = np.vstack((xHen20, yHen20))
    np.save('data_' + str(snapnum) + '_2020_SFRF', data)

# Testing the saved data #
snapnums = [11, 12, 13, 15, 17, 19]
for snapnum in snapnums:
    Hen15 = np.load('data_' + str(snapnum) + '_2015_SFRF.npy')
    Hen20 = np.load('data_' + str(snapnum) + '_2020_SFRF.npy')
    
    plt.close()
    f, ax = plt.subplots(1, figsize=(10, 7.5))
    plt.xlim(-1.0, 3.0)
    plt.ylim(-8.0, -1.0)
    
    plt.ylabel(r'$\mathrm{log_{10}}(\phi [\mathrm{Mpc^{-3}} \mathrm{log_{10}}(SFR^{-1})])$', fontsize=16)
    plt.xlabel(r'$\mathrm{log_{10}}(\mathrm{SFR}[M_{\odot}yr^{-1}])$', fontsize=16)
    plt.tick_params(direction='in', which='both', top='on', right='on')
    plt.plot(Hen15[0], np.log10(Hen15[1]), color='red', linestyle='dotted', label='Hen15')
    plt.plot(Hen20[0], np.log10(Hen20[1]), color='blue', linestyle='dotted', label='Hen20')
    plt.legend(loc=1)
    plt.text(0.0, 0.95, 'snapshot = ' + str(snapnum), fontsize=16, transform=ax.transAxes)
    plt.savefig('SFRF_' + str(snapnum) + date + '1.png', bbox_inches='tight')