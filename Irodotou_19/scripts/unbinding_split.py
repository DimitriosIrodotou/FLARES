import pygad as pg
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import glob
import math
import matplotlib
import astropy.constants as c
import astropy.units as u
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.cm as cm

#model = 'em20eff'
model = '0002eff'
#model = '001eff'
#model = '002eff'
#model = '02eff'
#model = '02noJ'
#model = '05eff'
#model = 'DelayedSF'

AllPlots = 'AllPlots/'

# Custom colormap
tabcols = ["tab:pink","tab:purple", "tab:cyan"]
cm.register_cmap(cmap=LinearSegmentedColormap.from_list("tabcmap", tabcols))
BGYcols = ["blue", "cyan", "limegreen", "yellow"]
cm.register_cmap(cmap=LinearSegmentedColormap.from_list("BGYcmap", BGYcols))

if model == 'em20eff':
    snappath = '../dwarf_compact_em20eff/snaps_em20eff/'
    snapstem = 'snap_em20eff_'
    modelnum = 'em20'
    effstr = '$\sim$ 0%'
    outpath='/freya/ptmp/mpa/jmhislop/sphgal_ketju/SCRIPTS_PLOTS/dwarf_compact_em20eff/'
    sf_outfile = 'sfr_em20eff/'
    histpath = 'hist_em20eff/'
    modelcol = 'tab:green'

elif model == '0002eff':
    snappath = '../dwarf_compact_0002eff/snaps_0002eff/'
    snapstem = 'snap_0002eff_'
    modelnum = '0002'
    effstr = '0.2%'
    outpath='/freya/ptmp/mpa/jmhislop/sphgal_ketju/SCRIPTS_PLOTS/dwarf_compact_0002eff/'
    sf_outfile = 'sfr_0002eff/'
    histpath = 'hist_0002eff/'
    virialpath = 'virialanalysis_0002/'
    propertiespath = 'clusterproperties_0002/'
    modelcol = 'tab:cyan'

elif model == '001eff':
    snappath = '../dwarf_compact_001eff/snaps_001eff/'
    snapstem = 'snap_001eff_'
    modelnum = '001'
    effstr = '1%'
    outpath='/freya/ptmp/mpa/jmhislop/sphgal_ketju/SCRIPTS_PLOTS/dwarf_compact_001eff/'
    sf_outfile = 'sfr_001eff/'
    histpath = 'hist_001eff/'
    virialpath = 'virialanalysis_001/'
    propertiespath = 'clusterproperties_001/'
    modelcol = 'tab:purple'

elif model == '002eff':
    snappath = '../dwarf_compact_002eff/snaps/outputsnaps/'
    snapstem = 'snap_test__'
    modelnum = '002'
    effstr = '2%'
    outpath='/freya/ptmp/mpa/jmhislop/sphgal_ketju/SCRIPTS_PLOTS/dwarf_compact_002eff/'
    sf_outfile = 'sfr_002eff/'
    histpath = 'hist_002eff/'
    virialpath = 'virialanalysis_002/'
    propertiespath = 'clusterproperties_002/'
    modelcol = 'tab:orange'

elif model == '02eff':
    snappath = '../dwarf_compact_02eff/snaps_02eff/'
    snapstem = 'snap_02eff_'
    modelnum = '02'
    effstr = '20%'
    outpath='/freya/ptmp/mpa/jmhislop/sphgal_ketju/SCRIPTS_PLOTS/dwarf_compact_02eff/'
    sf_outfile = 'sfr_02eff/'
    histpath = 'hist_02eff/'
    virialpath = 'virialanalysis_02/'
    propertiespath = 'clusterproperties_02/'
    modelcol = 'tab:pink'
    colors = ["white", modelcol]
    cm.register_cmap(cmap=LinearSegmentedColormap.from_list("modelcmap", colors))
    colors_black = [modelcol, "black"]
    cm.register_cmap(cmap=LinearSegmentedColormap.from_list("modelcmap_black", colors_black))

elif model == '02noJ':
    snappath = '../dwarf_compact_02eff_noJ/snaps_noJ_02eff/'
    snapstem = 'snap_noJ_02eff_'
    modelnum = '02noJ'
    effstr = '20%'
    outpath='/freya/ptmp/mpa/jmhislop/sphgal_ketju/SCRIPTS_PLOTS/dwarf_compact_02eff_noJ/'
    sf_outfile = 'sfr_02eff_noJ/'
    histpath = 'hist_02eff_noJ/'
    virialpath = 'virialanalysis_02_noJ/'
    modelcol = 'tab:pink'

elif model == '05eff':
    snappath = '../dwarf_compact_05eff/snaps_05eff/'
    snapstem = 'snap_05eff_'
    modelnum = '05'
    effstr = '50%'
    outpath='/freya/ptmp/mpa/jmhislop/sphgal_ketju/SCRIPTS_PLOTS/dwarf_compact_05eff/'
    sf_outfile = 'sfr_05eff/'
    histpath = 'hist_05eff/'
    virialpath = 'virialanalysis_05/'
    propertiespath = 'clusterproperties_05/'
    modelcol = 'tab:red'

elif model == 'DelayedSF':
    snappath = '/ptmp/mpa/uli/baby_dwarf_delayed_sfr_new/output_delay_instant/'
    snapstem = 'snap_'
    modelnum = 'dSF'
    effstr = 'delayedSF'
    outpath='/freya/ptmp/mpa/jmhislop/sphgal_ketju/SCRIPTS_PLOTS/delayedSF/'
    sf_outfile = 'sfr_dSF/'
    histpath = 'hist_dSF/'
    virialpath = 'virialanalysis_dSF/'
    modelcol = 'tab:red'

else:
    print('Model name not recognised')

nbins = 5
startsnap = 240
endsnap = 340
snapstep = 20

snaps_ = np.arange(startsnap, endsnap+snapstep, snapstep)

def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

NSnaps = int(round_down((endsnap-startsnap)/snapstep))

# Define empty arrays for things we need
maxFoF = 1000
maxNmembers = 5000

# Define arrays
s_time = np.zeros([NSnaps+1])
sub_particleN = np.zeros([NSnaps+1, maxFoF])
sub_mass_stars = np.zeros([NSnaps+1,maxFoF])
sub_avage_stars = np.zeros([NSnaps+1,maxFoF])
sub_time_ind_stars = np.zeros([NSnaps+1, maxFoF, maxNmembers])
sub_mass_ind_stars = np.zeros([NSnaps+1, maxFoF, maxNmembers])
sub_pos_stars = np.zeros([NSnaps+1,maxFoF,maxNmembers,3])

sub_mass_gas = np.zeros([NSnaps+1,maxFoF])
sub_avage_gas = np.zeros([NSnaps+1,maxFoF])

nbins = 10
hist_stars = np.zeros([NSnaps+1, nbins])
bin_centres_stars = np.zeros([NSnaps+1, nbins])
bin_widths_stars = np.zeros([NSnaps+1, nbins])

#hist_gas = np.zeros([NSnaps+1, nbins])
#bin_centres_gas = np.zeros([NSnaps+1, nbins])
#bin_widths_gas = np.zeros([NSnaps+1, nbins])

hist_agelim_stars = np.zeros([NSnaps+1, nbins])
bin_centres_agelim_stars = np.zeros([NSnaps+1, nbins])
bin_widths_agelim_stars = np.zeros([NSnaps+1, nbins])

#hist_agelim_gas = np.zeros([NSnaps+1, nbins])
#bin_centres_agelim_gas = np.zeros([NSnaps+1, nbins])
#bin_widths_agelim_gas = np.zeros([NSnaps+1, nbins])

hist_midagelim_stars = np.zeros([NSnaps+1, nbins])
bin_centres_midagelim_stars = np.zeros([NSnaps+1, nbins])
bin_widths_midagelim_stars = np.zeros([NSnaps+1, nbins])

hist_oldagelim_stars = np.zeros([NSnaps+1, nbins])
bin_centres_oldagelim_stars = np.zeros([NSnaps+1, nbins])
bin_widths_oldagelim_stars = np.zeros([NSnaps+1, nbins])

#N_FoF_stars = np.zeros([NSnaps+1])
#N_FoF_gas = np.zeros([NSnaps+1])

#CoM_stars = np.zeros([NSnaps+1, maxFoF, 3])
#CoM_gas = np.zeros([NSnaps+1, maxFoF, 3])

x_min_stars = np.zeros([NSnaps+1, maxFoF])
x_max_stars = np.zeros([NSnaps+1, maxFoF])
y_min_stars = np.zeros([NSnaps+1, maxFoF])
y_max_stars = np.zeros([NSnaps+1, maxFoF])
z_min_stars = np.zeros([NSnaps+1, maxFoF])
z_max_stars = np.zeros([NSnaps+1, maxFoF])

gasmass_inSCs = np.zeros([NSnaps+1, maxFoF])
stellarmass_inSCs = np.zeros([NSnaps+1, maxFoF])

# Calculate velocity of CoM
# Subtract v_CoM from velocities of each particle
# Sum up 1/2 m_i v_i**2

# Convert Masses: Msun -> kg
convMass = u.Msun.to(u.kg, 1)
# Need to convert G: m -> km
siG = c.G.to(u.km**3/(u.kg*u.s**2)).value
# Need to convert Radii: pc -> km
convRad = u.pc.to(u.km, 1)

# Try in astro units
#4.30091(25)×10−3	pc⋅M⊙–1⋅(km/s)2
astroG = c.G.to(((u.pc/u.Msun))*(u.km/u.s)**2).value

names = glob.glob(outpath+virialpath+'Tot_Energy_stars_'+modelnum+'eff_*')
#names = []


for j in np.arange(startsnap, endsnap+1, snapstep):
#for j in np.arange(300, 301, snapstep):

    #if outpath+virialpath+'Tot_Energy_stars_'+modelnum+'eff_'+'snap{:03d}'.format(j)+'.npy' in names:
        #print('Snap '+str(j)+' Already Calculated')
        #continue

    print('Analysing Snap ',str(j))

    FoF_stars = np.load(outpath+'FoF_stars_'+modelnum+'/FoF_stars_'+modelnum+'_'+'snap{:03d}'.format(j)+'.npy')
    # Number of FoF groups
    N_FoF_stars = np.load(outpath+'N_FoF_stars_'+modelnum+'/N_FoF_stars_'+modelnum+'_'+'snap{:03d}'.format(j)+'.npy')
    CoM_stars = np.load(outpath+'CoM_stars_'+modelnum+'/CoM_stars_'+modelnum+'_'+'snap{:03d}'.format(j)+'.npy')

    s = pg.Snapshot(snappath+snapstem+'{:03d}'.format(j)+'.hdf5')
    s.to_physical_units()
    s_time = s.time

    _sub_particleN = np.zeros([int(N_FoF_stars)])
    _sub_mass_stars = np.zeros([int(N_FoF_stars)])
    _sub_avage_stars = np.zeros([int(N_FoF_stars)])
    _sub_time_ind_stars = np.zeros([int(N_FoF_stars), maxNmembers])
    _sub_mass_ind_stars = np.zeros([int(N_FoF_stars), maxNmembers])
    _sub_pos_stars = np.zeros([int(N_FoF_stars),maxNmembers,3])

    # Sort the stars in each fof group by radius compute a cumulative mass profile and take the radius at half mass
    # ONE SNAP ONLY
    # Shrinking sphere analysis
    # Take a rough sphere around the cluster (estimated from max radii of cluster) and calculate CoM. Reduce size of sphere and recalculate CoM. This will find the centre of the cluster, can use CM = numpy.average(Cpos, axis=0, weights=Cmass)
    # To find r_1/2 : for r in np.arange(0, max(R), max(R)/10000?): M(<r) = M[np.where(R<r)]. If M(<r)<=M_1/2, stop.
    r_hm_stars = np.zeros([int(N_FoF_stars)])
    half_mass = np.zeros([int(N_FoF_stars)])
    cum_mass = np.zeros([int(N_FoF_stars)])
    rho_hm = np.zeros([int(N_FoF_stars)])
    surfacedens_tot = np.zeros([int(N_FoF_stars)])
    max_rad = np.zeros([int(N_FoF_stars)])
    r_90_stars = np.zeros([int(N_FoF_stars)])
    surfacedens_hm = np.zeros([int(N_FoF_stars)])
    #r_hm_gas = np.zeros([NSnaps,int(max(N_FoF_gas))])
    rad_stars = np.zeros([int(N_FoF_stars), maxNmembers])

    _sub_mass_ind_stars = np.zeros([int(N_FoF_stars), maxNmembers])
    _sub_vel_ind_stars = np.zeros([int(N_FoF_stars), maxNmembers,3])
    _sub_totvel_ind_stars = np.zeros([int(N_FoF_stars), maxNmembers])
    _sub_particleN = np.zeros([int(N_FoF_stars)])
    vel_mean = np.zeros([int(N_FoF_stars),3])
    KE_ind_stars = np.zeros([int(N_FoF_stars), maxNmembers])
    KE_normed_ind_stars = np.zeros([int(N_FoF_stars), maxNmembers])
    U_ind_stars = np.zeros([int(N_FoF_stars), maxNmembers, maxNmembers])
    U_tot_ind_stars = np.zeros([int(N_FoF_stars), maxNmembers])
    vels_ind_stars = np.zeros([int(N_FoF_stars), maxNmembers])
    tot_normed_vels_ind = np.zeros([int(N_FoF_stars), maxNmembers])
    normed_vels_ind = np.zeros([int(N_FoF_stars), maxNmembers,3])
    KE_tot_stars = np.zeros([int(N_FoF_stars)])
    KE_normed_tot_stars = np.zeros([int(N_FoF_stars)])
    U_tot_stars = np.zeros([int(N_FoF_stars)])
    vel_disp = np.zeros([int(N_FoF_stars)])
    index_sorted = np.zeros([int(N_FoF_stars), maxNmembers], dtype=np.int16)
    sorted_rad = np.zeros([int(N_FoF_stars), maxNmembers])
    sorted_masses = np.zeros([int(N_FoF_stars), maxNmembers])

    Tot_Energy_ind_stars = np.zeros([int(N_FoF_stars), maxNmembers])
    Tot_Energy_stars = np.zeros([int(N_FoF_stars)])

    # Arrays for BOUND clusters #

    bound_sub_particleN = np.zeros([int(N_FoF_stars)])
    bound_sub_mass_stars = np.zeros([int(N_FoF_stars)])
    bound_sub_avage_stars = np.zeros([int(N_FoF_stars)])
    bound_sub_time_ind_stars = np.zeros([int(N_FoF_stars), maxNmembers])
    bound_sub_mass_ind_stars = np.zeros([int(N_FoF_stars), maxNmembers])
    bound_sub_pos_stars = np.zeros([int(N_FoF_stars),maxNmembers,3])

    bound_r_hm_stars = np.zeros([int(N_FoF_stars)])
    bound_half_mass = np.zeros([int(N_FoF_stars)])
    bound_cum_mass = np.zeros([int(N_FoF_stars)])
    bound_rho_hm = np.zeros([int(N_FoF_stars)])
    bound_surfacedens_tot = np.zeros([int(N_FoF_stars)])
    bound_max_rad = np.zeros([int(N_FoF_stars)])
    bound_r_90_stars = np.zeros([int(N_FoF_stars)])
    bound_surfacedens_hm = np.zeros([int(N_FoF_stars)])
    #r_hm_gas = np.zeros([NSnaps,int(max(N_FoF_gas))])
    bound_rad_stars = np.zeros([int(N_FoF_stars), maxNmembers])

    bound_sub_mass_ind_stars = np.zeros([int(N_FoF_stars), maxNmembers])
    bound_sub_vel_ind_stars = np.zeros([int(N_FoF_stars), maxNmembers,3])
    bound_sub_totvel_ind_stars = np.zeros([int(N_FoF_stars), maxNmembers])
    bound_sub_particleN = np.zeros([int(N_FoF_stars)])
    bound_vel_mean = np.zeros([int(N_FoF_stars),3])
    bound_KE_ind_stars = np.zeros([int(N_FoF_stars), maxNmembers])
    bound_KE_normed_ind_stars = np.zeros([int(N_FoF_stars), maxNmembers])
    bound_U_ind_stars = np.zeros([int(N_FoF_stars), maxNmembers, maxNmembers])
    bound_U_tot_ind_stars = np.zeros([int(N_FoF_stars), maxNmembers])
    bound_vels_ind_stars = np.zeros([int(N_FoF_stars), maxNmembers])
    bound_tot_normed_vels_ind = np.zeros([int(N_FoF_stars), maxNmembers])
    bound_normed_vels_ind = np.zeros([int(N_FoF_stars), maxNmembers,3])
    bound_KE_tot_stars = np.zeros([int(N_FoF_stars)])
    bound_KE_normed_tot_stars = np.zeros([int(N_FoF_stars)])
    bound_U_tot_stars = np.zeros([int(N_FoF_stars)])
    bound_vel_disp = np.zeros([int(N_FoF_stars)])
    bound_index_sorted = np.zeros([int(N_FoF_stars), maxNmembers], dtype=np.int16)
    bound_sorted_rad = np.zeros([int(N_FoF_stars), maxNmembers])
    bound_sorted_masses = np.zeros([int(N_FoF_stars), maxNmembers])

    # Unbinding arrays
    bound_Tot_Energy_ind_stars = np.zeros([int(N_FoF_stars), maxNmembers])
    bound_Tot_Energy_stars = np.zeros([int(N_FoF_stars)])
    bound_Tot_U_stars = np.zeros([int(N_FoF_stars)])
    bound_Tot_normed_KE_stars = np.zeros([int(N_FoF_stars)])
    bound_KE = np.zeros([int(N_FoF_stars), maxNmembers])
    energy_index = np.zeros([int(N_FoF_stars),maxNmembers], dtype=np.int16)
    energy_sorted = np.zeros([int(N_FoF_stars),maxNmembers])
    boundstarsind = np.zeros([int(N_FoF_stars),maxNmembers], dtype=np.int16)
    nonboundstarsind = np.zeros([int(N_FoF_stars),maxNmembers], dtype=np.int16)

    bound_U_tot_ind_stars = np.zeros([int(N_FoF_stars), maxNmembers])
    bound_U_tot_stars = np.zeros([int(N_FoF_stars)])
    N_members = np.zeros([int(N_FoF_stars)])
    bound_members = np.zeros([int(N_FoF_stars), maxNmembers], dtype=bool)


    # Loop through each cluster
    for i in np.arange(0, int(N_FoF_stars), 1):
    #for i in np.arange(0, 1, 1):
        print('Analysing cluster '+str(i))
        _sub_particleN[i] = len(s.stars[FoF_stars==i]['mass'])
        _sub_mass_ind_stars[i, :int(_sub_particleN[i])] = s.stars[FoF_stars==i]['mass']
        _sub_mass_stars[i] = s.stars[FoF_stars==i]['mass'].sum()
        # Average age of each FoF group
        _sub_avage_stars[i] = np.average(s.time - s.stars[FoF_stars==i]['form_time'])
        # 3 coordinate position of each star within each star cluster
        _sub_pos_stars[i, :int(_sub_particleN[i]), :] = s.stars[FoF_stars==i]['pos']
        # Convert from kpc to pc
        rad_stars[i,:int(_sub_particleN[i])] = 10**3*np.sqrt((np.absolute(CoM_stars[i, 0]-_sub_pos_stars[i , :int(_sub_particleN[i]), 0]))**2 + (np.absolute(CoM_stars[i, 1]-_sub_pos_stars[i , :int(_sub_particleN[i]), 1]))**2 + (np.absolute(CoM_stars[i, 2]-_sub_pos_stars[i , :int(_sub_particleN[i]), 2]))**2)
        index_sorted[i,:int(_sub_particleN[i])] = np.argsort(rad_stars[i,:int(_sub_particleN[i])])

        sorted_rad[i,:int(_sub_particleN[i])] = rad_stars[i,:int(_sub_particleN[i])][index_sorted[i,:int(_sub_particleN[i])]]
        sorted_masses[i,:int(_sub_particleN[i])] = _sub_mass_ind_stars[i,:int(_sub_particleN[i])][index_sorted[i,:int(_sub_particleN[i])]]
        cum_mass = np.cumsum(sorted_masses[i,:int(_sub_particleN[i])])
        if (cum_mass[-1] - _sub_mass_stars[i]) > 0.1:
            print('cumulative mass and total mass not the same!')
            continue
        max_rad[i] = sorted_rad[i,:int(_sub_particleN[i])][-1]
        half_mass[i] = np.sum(sorted_masses[i,:int(_sub_particleN[i])])/2
        r_hm_stars[i] = sorted_rad[i,:int(_sub_particleN[i])][np.where(cum_mass>half_mass[i])[0][0]]
        r_90_stars[i] = sorted_rad[i,:int(_sub_particleN[i])][np.where(cum_mass>0.9*_sub_mass_stars[i])[0][0]]
        surfacedens_hm[i] = half_mass[i]/(np.pi*r_hm_stars[i]**2)
        rho_hm[i] = half_mass[i]/((4/3)*np.pi*r_hm_stars[i]**3)
        surfacedens_tot[i] = _sub_mass_stars[i]/(np.pi*max_rad[i]**2)

        # Now these are calculated, go back to using the UNORDERED arrays (aka in order of what comes out of FoF)
        # This allows bound_members to be a mask that can be applied to the output directly from FoF in future

        # Calculating Kinetic energy: 1/2mv^2
        _sub_vel_ind_stars[i, :int(_sub_particleN[i]),:] = s.stars[FoF_stars==i]['vel']
        vels_ind_stars[i, :int(_sub_particleN[i])] = np.sqrt((_sub_vel_ind_stars[i,:int(_sub_particleN[i]),:] ** 2).sum(axis=1))
        vel_mean[i,:] = np.average(_sub_vel_ind_stars[i, :int(_sub_particleN[i]),:], axis=0)
        normed_vels_ind[i, :int(_sub_particleN[i]),:] = _sub_vel_ind_stars[i, :int(_sub_particleN[i]),:] - vel_mean[i,:]
        tot_normed_vels_ind[i, :int(_sub_particleN[i])] = np.sqrt((normed_vels_ind[i,:int(_sub_particleN[i]),:] ** 2).sum(axis=1))

        for ii in np.arange(0, int(_sub_particleN[i]), 1):
            # Units: Msun * (km/s)**2
            KE_normed_ind_stars[i, ii] = 0.5*(_sub_mass_ind_stars[i,ii])*(tot_normed_vels_ind[i,ii]**2)
            KE_ind_stars[i, ii] = 0.5*(_sub_mass_ind_stars[i,ii])*(vels_ind_stars[i,ii]**2)
        KE_normed_tot_stars[i] = np.sum(KE_normed_ind_stars[i,:])
        KE_tot_stars[i] = np.sum(KE_ind_stars[i,:])

        #for a in np.arange(0, int(_sub_particleN[i]), 1):
        #    for b in np.arange(a+1, int(_sub_particleN[i]), 1):
        #        U_ind_stars[i,a,b] = -(astroG*_sub_mass_ind_stars[i,a]*_sub_mass_ind_stars[i,b])/(abs(rad_stars[i,a]-rad_stars[i,b]))
        #        U_ind_stars[i,b,a] = U_ind_stars[i,a,b]
        #U_tot_ind_stars[i, :int(_sub_particleN[i])] = np.sum(U_ind_stars[i, :int(_sub_particleN[i]), :int(_sub_particleN[i])], axis=0)
        #U_tot_stars[i] = np.sum(U_tot_ind_stars[i,:int(_sub_particleN[i])])
        l=0
        m=0
        for a in np.arange(0, int(_sub_particleN[i]), 1):
            for b in np.arange(a+1, int(_sub_particleN[i]), 1):
                if (abs(sorted_rad[i,a]-sorted_rad[i,b]))<0.1:
                    l+=1
                    U_ind_stars[i,a,b] = -(astroG*_sub_mass_ind_stars[i,a]*_sub_mass_ind_stars[i,b])/(np.sqrt((rad_stars[i,a]-rad_stars[i,b])**2+0.1**2))
                    U_ind_stars[i,b,a] = U_ind_stars[i,a,b]
                else:
                    m+=1
                    U_ind_stars[i,a,b] = -(astroG*_sub_mass_ind_stars[i,a]*_sub_mass_ind_stars[i,b])/(abs(rad_stars[i,a]-rad_stars[i,b]))
                    U_ind_stars[i,b,a] = U_ind_stars[i,a,b]
        U_tot_ind_stars[i, :int(_sub_particleN[i])] = np.sum(U_ind_stars[i, :int(_sub_particleN[i]), :int(_sub_particleN[i])], axis=0)
        U_tot_stars[i] = np.sum(U_tot_ind_stars[i,:int(_sub_particleN[i])])

        # Calculating potential energy: Assuming spherical symmetry, U = -3/5 GM^2/R
        # Units: pc/Msun * (km/s)**2 * Msun**2 * pc^-1
        #U_tot_stars[i] = (-3/5)*astroG*_sub_mass_stars[i]**2/max_rad[i]

        # Calculate the velocity dispersion
        vel_disp[i] = np.std(normed_vels_ind[i, :int(_sub_particleN[i])])

        for ii in np.arange(0, int(_sub_particleN[i]), 1):
            Tot_Energy_ind_stars[i,ii] = 2*KE_normed_ind_stars[i,ii] + U_tot_ind_stars[i,ii]
        Tot_Energy_stars[i] = np.sum(Tot_Energy_ind_stars[i,:int(_sub_particleN[i])])

        ##################### NOW ADD IN FULL UNBINDING #####################

        for ii in np.arange(0, int(_sub_particleN[i]), 1):
            bound_Tot_Energy_ind_stars[i,ii] = 2*KE_normed_ind_stars[i,ii] + U_tot_ind_stars[i,ii]


        N_members[i] = int(_sub_particleN[i])
        bound_members[i, :int(_sub_particleN[i])] = True

        # indices of stars that are bound to the cluster
        b_ind = bound_Tot_Energy_ind_stars[i,:int(_sub_particleN[i])]<=0
        bound_index = bound_Tot_Energy_ind_stars[i,:int(_sub_particleN[i])][b_ind]
        unbound_index = bound_Tot_Energy_ind_stars[i,:int(_sub_particleN[i])][~b_ind]
        bound_members[i,:int(_sub_particleN[i])][~b_ind] = False

        # Calculate new potentials

        bound_U_ind_stars = np.copy(U_ind_stars)
        bound_KE_normed_ind_stars = np.copy(KE_normed_ind_stars)

        bound_U_ind_stars[i, :int(_sub_particleN[i]), :int(_sub_particleN[i])][:, ~b_ind] = 0
        bound_U_ind_stars[i, :int(_sub_particleN[i]), :int(_sub_particleN[i])][~b_ind, :] = 0
        bound_KE_normed_ind_stars[i, :int(_sub_particleN[i])][~b_ind] = 0
        
        if len(np.where(bound_members[i,:]==True)[0])<35:
                bound_members[i,:int(_sub_particleN[i])] = False

        # Rather than recalculating potentials directly, just need to sum over the row/column with just the indices needed
        bound_U_tot_ind_stars[i, :int(_sub_particleN[i])] = np.sum(bound_U_ind_stars[i, :int(_sub_particleN[i]), :int(_sub_particleN[i])], axis=0)

        count = 0
        while len(np.where(bound_members[i,:]==True)[0])>35 and len(unbound_index)!=0:
            print('Count = ',count)
            # Calculate total energies
            for ii in np.arange(0, int(_sub_particleN[i]), 1):
                bound_Tot_Energy_ind_stars[i,ii] = 2*bound_KE_normed_ind_stars[i, :int(_sub_particleN[i])][ii] + bound_U_tot_ind_stars[i,:int(_sub_particleN[i])][ii]
            #print('TotEnergy = ',bound_Tot_Energy_ind_stars[i,:int(_sub_particleN[i])])
            # Find indices of bound stars
            b_ind = bound_Tot_Energy_ind_stars[i,:int(_sub_particleN[i])]<=0
            bound_index = bound_Tot_Energy_ind_stars[i,:int(_sub_particleN[i])][b_ind]
            unbound_index = bound_Tot_Energy_ind_stars[i,:int(_sub_particleN[i])][~b_ind]
            bound_members[i,:int(_sub_particleN[i])][~b_ind] = False

            #print('bound_members = ',bound_members[i,:int(_sub_particleN[i])])

            # Calculate new potentials
            bound_U_ind_stars[i, :int(_sub_particleN[i]), :int(_sub_particleN[i])][~b_ind, :] = 0
            bound_U_ind_stars[i, :int(_sub_particleN[i]), :int(_sub_particleN[i])][:, ~b_ind] = 0

            bound_U_tot_ind_stars[i, :int(_sub_particleN[i])] = np.sum(bound_U_ind_stars[i, :int(_sub_particleN[i]), :int(_sub_particleN[i])], axis=0)
            # Kinetic energies stay the same but set nonbound energies to zero
            bound_KE_normed_ind_stars[i, :int(_sub_particleN[i])][~b_ind] = 0


            if len(np.where(bound_members[i,:]==True)[0])<35:
                bound_members[i,:int(_sub_particleN[i])] = False

            count+=1

        # Calculate total potential - can simply sum as nonbound members are set to zero
        bound_U_tot_stars[i] = np.sum(bound_U_tot_ind_stars[i,:int(_sub_particleN[i])])
        bound_KE_normed_tot_stars[i] = np.sum(bound_KE_normed_ind_stars[i,:])

        for ii in np.arange(0, int(_sub_particleN[i]), 1):
            bound_Tot_Energy_ind_stars[i,ii] = 2*bound_KE_normed_ind_stars[i,ii] + bound_U_tot_ind_stars[i,ii]

        bound_Tot_Energy_stars[i] = np.sum(bound_Tot_Energy_ind_stars[i,:int(_sub_particleN[i])])



        # UNBINDING FINISHED

        # Recalculate everything but using only the bound members of each cluster
        # Just need to mask with [bound_members[i]] which has a length of 5000
        # bound_members[i, :int(_sub_particleN[i])]
        bound_members_red = bound_members[i, :int(_sub_particleN[i])]
        # CHECK shape of bound_members_red
        bound_sub_particleN[i] = len(bound_members_red)
        nbound = len(np.where(bound_members_red!=False)[0])
        if nbound==0:
            print('Cluster '+str(i)+' not bound')
            continue
        else:
            bound_sub_mass_ind_stars[i, :int(nbound)] = s.stars[FoF_stars==i]['mass'][bound_members_red]
            bound_sub_mass_stars[i] = s.stars[FoF_stars==i]['mass'][bound_members_red].sum()
            # Average age of each FoF group
            bound_sub_avage_stars[i] = np.average(s.time - s.stars[FoF_stars==i]['form_time'][bound_members_red])
            # 3 coordinate position of each star within each star cluster
            bound_sub_pos_stars[i, :int(nbound), :] = s.stars[FoF_stars==i]['pos'][bound_members_red]
            # Convert from kpc to pc
            bound_rad_stars[i,:int(nbound)] = 10**3*np.sqrt((np.absolute(CoM_stars[i, 0]-bound_sub_pos_stars[i , :int(nbound), 0]))**2 + (np.absolute(CoM_stars[i, 1]-bound_sub_pos_stars[i , :int(nbound), 1]))**2 + (np.absolute(CoM_stars[i, 2]-_sub_pos_stars[i , :int(nbound), 2]))**2)
            bound_index_sorted[i,:int(nbound)] = np.argsort(bound_rad_stars[i,:int(nbound)])

            bound_sorted_rad[i,:int(nbound)] = bound_rad_stars[i,:int(nbound)][bound_index_sorted[i,:int(nbound)]]
            bound_sorted_masses[i,:int(nbound)] = bound_sub_mass_ind_stars[i,:int(nbound)][bound_index_sorted[i,:int(nbound)]]
            bound_cum_mass = np.cumsum(bound_sorted_masses[i,:int(nbound)])
            if (bound_cum_mass[-1] - bound_sub_mass_stars[i]) > 0.1:
                print('cumulative mass and total mass not the same!')
                continue
            bound_max_rad[i] = bound_sorted_rad[i,:int(nbound)][-1]
            bound_half_mass[i] = np.sum(bound_sorted_masses[i,:int(nbound)])/2
            bound_r_hm_stars[i] = bound_sorted_rad[i,:int(nbound)][np.where(bound_cum_mass>bound_half_mass[i])[0][0]]
            bound_r_90_stars[i] = bound_sorted_rad[i,:int(nbound)][np.where(bound_cum_mass>0.9*bound_sub_mass_stars[i])[0][0]]
            bound_surfacedens_hm[i] = bound_half_mass[i]/(np.pi*bound_r_hm_stars[i]**2)
            bound_rho_hm[i] = bound_half_mass[i]/((4/3)*np.pi*bound_r_hm_stars[i]**3)
            bound_surfacedens_tot[i] = bound_sub_mass_stars[i]/(np.pi*bound_max_rad[i]**2)

    print('Saving unbound arrays')
    # Save arrays without unbinding applied
    np.save(outpath+virialpath+'KE_normed_tot_stars_'+modelnum+'eff_'+'snap{:03d}'.format(j)+'.npy', KE_normed_tot_stars)
    np.save(outpath+virialpath+'U_tot_stars_'+modelnum+'eff_'+'snap{:03d}'.format(j)+'.npy', U_tot_stars)
    np.save(outpath+virialpath+'Tot_Energy_stars_'+modelnum+'eff_'+'snap{:03d}'.format(j)+'.npy', Tot_Energy_stars)

    np.save(outpath+propertiespath+'sub_mass_ind_stars_'+modelnum+'eff_'+'snap{:03d}'.format(j)+'.npy', _sub_mass_ind_stars)
    np.save(outpath+propertiespath+'sub_mass_stars_'+modelnum+'eff_'+'snap{:03d}'.format(j)+'.npy', _sub_mass_stars)
    np.save(outpath+propertiespath+'sub_avage_stars_'+modelnum+'eff_'+'snap{:03d}'.format(j)+'.npy', _sub_avage_stars)
    np.save(outpath+propertiespath+'r_hm_stars_'+modelnum+'eff_'+'snap{:03d}'.format(j)+'.npy', r_hm_stars)
    np.save(outpath+propertiespath+'surfacedens_hm_'+modelnum+'eff_'+'snap{:03d}'.format(j)+'.npy', surfacedens_hm)
    np.save(outpath+propertiespath+'surfacedens_tot_'+modelnum+'eff_'+'snap{:03d}'.format(j)+'.npy', surfacedens_tot)

    print('Saving bound arrays')
    # Save same arrays with unbinding applied
    #bound_ind = np.where(Tot_Energy_stars<=0)[0]
    #nonbound_ind = np.where(Tot_Energy_stars>0)[0]
    np.save(outpath+virialpath+'bound_KE_normed_tot_stars'+modelnum+'eff_'+'snap{:03d}'.format(j)+'.npy', bound_KE_normed_tot_stars)
    np.save(outpath+virialpath+'bound_U_tot_stars'+modelnum+'eff_'+'snap{:03d}'.format(j)+'.npy', bound_U_tot_stars)
    np.save(outpath+virialpath+'Bound_Tot_Energy_stars_'+modelnum+'eff_'+'snap{:03d}'.format(j)+'.npy', bound_Tot_Energy_stars)
    np.save(outpath+virialpath+'Bound_time_bound_'+modelnum+'eff_'+'snap{:03d}'.format(j)+'.npy', s_time)
    np.save(outpath+virialpath+'bound_members_'+modelnum+'eff_'+'snap{:03d}'.format(j)+'.npy', bound_members)
    np.save(outpath+virialpath+'N_members_'+modelnum+'eff_'+'snap{:03d}'.format(j)+'.npy', N_members)


    np.save(outpath+propertiespath+'bound_sub_mass_ind_stars_'+modelnum+'eff_'+'snap{:03d}'.format(j)+'.npy', bound_sub_mass_ind_stars)
    np.save(outpath+propertiespath+'bound_sub_mass_stars_'+modelnum+'eff_'+'snap{:03d}'.format(j)+'.npy', bound_sub_mass_stars)
    np.save(outpath+propertiespath+'bound_sub_avage_stars_'+modelnum+'eff_'+'snap{:03d}'.format(j)+'.npy', bound_sub_avage_stars)
    np.save(outpath+propertiespath+'bound_r_hm_stars_'+modelnum+'eff_'+'snap{:03d}'.format(j)+'.npy', bound_r_hm_stars)
    np.save(outpath+propertiespath+'bound_surfacedens_hm_'+modelnum+'eff_'+'snap{:03d}'.format(j)+'.npy', bound_surfacedens_hm)
    np.save(outpath+propertiespath+'bound_surfacedens_tot_'+modelnum+'eff_'+'snap{:03d}'.format(j)+'.npy', bound_surfacedens_tot)


    #np.save(outpath+virialpath+'nonbound_ind_'+modelnum+'eff_'+'snap{:03d}'.format(j)+'.npy', nonbound_ind)


    # Remove after testing
    #np.save(outpath+virialpath+'U_ind_stars'+modelnum+'eff_'+'snap{:03d}'.format(j)+'.npy', U_ind_stars)


#'''

