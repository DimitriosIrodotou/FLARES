import time
import numpy as np
import matplotlib.pyplot as plt

date = time.strftime("%d\%m\%y\%H%M")

# Read HWT15 data for redshift 5 to 10 #s
snapnums = [11, 12, 13, 15, 17, 19]
redshifts = [9.72, 8.93, 8.22, 6.97, 5.92, 5.03]
i = 0
for snapnum in snapnums:
    # Read HWT15 data for redshift 5 to 10 #
    HWT15 = np.loadtxt('snap_' + str(snapnum) + '_2015.txt',
                       dtype={'names': ('type', 'stellar_mass', 'sfr'), 'formats': (np.int, np.float, np.float)}, delimiter=',', skiprows=12)
    
    # Read HWT15 MRII data for redshift 5 to 10 #
    # HWT15_MRII = np.loadtxt('snap_' + str(snapnum) + '_2015_MRII.txt',
    #                         dtype={'names': ('type', 'stellar_mass', 'sfr'), 'formats': (np.int, np.float, np.float)}, delimiter=',', skiprows=12)
    
    # Read HYF20 data for redshift 5 to 10 #
    HWT20 = np.loadtxt('snap_' + str(snapnum) + '_2020.txt',
                       dtype={'names': ('type', 'stellar_mass', 'sfr'), 'formats': (np.int, np.float, np.float)}, delimiter=',', skiprows=12)
    
    # Put into bins and normalise to number per unit volume (Mpc/h) per dex
    hubble = 0.673  # [dimensionless]
    boxside = 480.28 / hubble  # [Mpc]
    boxside_MRII = 96.0558 / hubble  # [Mpc]
    
    firstFile = 0
    lastFile = 511
    maxFile = 512
    binperdex = 5
    xrange = np.array([7.8, 12.0])
    nbin = (xrange[1] - xrange[0]) * binperdex
    
    offset = 10 - np.log10(hubble)
    log10StellarMass = np.log10(HWT15['stellar_mass']) + offset
    log10StellarMassObs = log10StellarMass + np.random.randn(len(HWT15['stellar_mass'])) * 0.08 * (1 + redshifts[i])
    
    nobj15, bins15, junk15 = plt.hist(log10StellarMass, bins=np.int(nbin), range=xrange, log=True)
    yHen15 = nobj15 * maxFile / ((lastFile - firstFile + 1) * boxside ** 3) * binperdex
    xHen15 = 0.5 * (bins15[:-1] + bins15[1:])
    data = np.vstack((xHen15, yHen15))
    np.save('data_' + str(snapnum) + '_2015', data)
    
    # nobj15_MRII, bins15_MRII, junk15_MRII = plt.hist(np.log10(HWT15_MRII['stellar_mass'] * 1e10 / hubble), bins=np.int(nbin), range=xrange, log=True)
    # yHen15_MRII = nobj15_MRII * maxFile / ((lastFile - firstFile + 1) * boxside_MRII ** 3) * binperdex
    # xHen15_MRII = 0.5 * (bins15[:-1] + bins15[1:])
    # data = np.vstack((xHen15_MRII, yHen15_MRII))
    # np.save('data_' + str(snapnum) + '_2015_MRII', data)
    
    nobj20, bins20, junk20 = plt.hist(np.log10(HWT20['stellar_mass'] * 1e10 / hubble), bins=np.int(nbin), range=xrange, log=True)
    yHen20 = nobj20 * maxFile / ((lastFile - firstFile + 1) * boxside ** 3) * binperdex
    xHen20 = 0.5 * (bins20[:-1] + bins20[1:])
    data = np.vstack((xHen20, yHen20))
    np.save('data_' + str(snapnum) + '_2020', data)
    
    i += 1
# Testing the saved data #
snapnums = [11, 12, 13, 15, 17, 19]
for snapnum in snapnums:
    Hen15 = np.load('data_' + str(snapnum) + '_2015.npy')
    # Hen15_MRII = np.load('SMF/data_' + str(snapnum) + '_2015_MRII.npy')
    Hen20 = np.load('data_' + str(snapnum) + '_2020.npy')
    
    plt.close()
    f, ax = plt.subplots(1, figsize=(10, 7.5))
    plt.ylim(-10, -1)
    plt.xlim(8, 12)
    
    plt.ylabel(r'$\mathrm{log_{10}(N / (dex / (Mpc)^{3})}$', fontsize=16)
    plt.xlabel(r'$\mathrm{log_{10}(M_{\bigstar} / M_\odot)}$', fontsize=16)
    plt.tick_params(direction='in', which='both', top='on', right='on')
    plt.plot(Hen15[0], np.log10(Hen15[1]), color='red', linestyle='dotted', label='Hen15')
    # plt.plot(Hen15_MRII[0], np.log10(Hen15_MRII[1]), color='green', linestyle='dotted', label='Hen15_MRII')
    plt.plot(Hen20[0], np.log10(Hen20[1]), color='blue', linestyle='dotted', label='Hen20')
    plt.legend(loc=1)
    plt.text(0.0, 0.95, 'snapshot = ' + str(snapnum), fontsize=16, transform=ax.transAxes)
    plt.savefig('SMF_' + str(snapnum) + date + '1.png', bbox_inches='tight')