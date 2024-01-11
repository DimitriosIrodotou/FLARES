import os
import sys
import h5py
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

from flares import flares

fl = flares.flares(fname='/cosma7/data/dp004/dc-love2/codes/flares/data/flares.h5')

halo = fl.halos[0]
_idx = 68  # 68 # index in master file for a quiescent galaxy

tag_idx = 5
# fl.print_graph_keys(halo,fl.tags[tag_idx])

# find progenitors in previous snapshot
# first load previous snap arrays
group_number = fl.load_dataset('GroupNumber', arr_type='Galaxy')
sgroup_number = fl.load_dataset('SubGroupNumber', arr_type='Galaxy')
mstar = fl.load_dataset('Mstar_aperture/Mstar_30', arr_type='Galaxy')
gmass = {tag:None for tag in fl.tags}  # generate gas masses (aperture) from particle data
# for tag in fl.tags:
#     _gmass = fl.get_particles('G_Mass', length_array='G_Length', halo=halo, tag=tag)
#     gmass[tag] = np.array([np.sum(_g['G_Mass']) for key, _g in _gmass.items()])

prog_idxs = [_idx]  # set main 'progenitor' to the index of choice initially
for tag_idx in [5, 4, 3, 2, 1]:
    print(fl.tags[tag_idx])
    print("Index,group,subgroup:", prog_idxs[0], group_number[halo][fl.tags[tag_idx]][prog_idxs[0]],
          sgroup_number[halo][fl.tags[tag_idx]][prog_idxs[0]])

    imass = mstar[halo][fl.tags[tag_idx]][prog_idxs[0]]
    print("log10 Mstar: %.2f" % np.log10(imass * 1e10))
    print("--------")

    output = fl.get_progenitors(group_number[halo][fl.tags[tag_idx]][prog_idxs[0]],
                                sgroup_number[halo][fl.tags[tag_idx]][prog_idxs[0]], halo, fl.tags[tag_idx],
                                properties=['prog_group_ids', 'prog_subgroup_ids', 'prog_stellar_masses',
                                            'prog_stellar_mass_contribution'])

    grp = output['prog_group_ids']
    sgrp = output['prog_subgroup_ids']
    _mstar = output['prog_stellar_masses']
    _mstar_con = output['prog_stellar_mass_contribution']

    # find indexes in master file
    prog_idxs = [
        np.where((group_number[halo][fl.tags[tag_idx - 1]] == _g) & (sgroup_number[halo][fl.tags[tag_idx - 1]] == _sg))[
            0] for _g, _sg in zip(grp, sgrp)]

    prog_idxs = np.array([_p for _p in prog_idxs if len(_p) > 0]).T[0]

    _mstar_con[_mstar_con == 0.] = np.nan

    with np.errstate(invalid='ignore'):
        print("Progenitors (%s):" % fl.tags[tag_idx - 1])
        print("Index,group,subgroup:", prog_idxs, group_number[halo][fl.tags[tag_idx - 1]][prog_idxs],
              sgroup_number[halo][fl.tags[tag_idx - 1]][prog_idxs])
        print("log10 Mstar:", np.log10(_mstar[:len(prog_idxs)] * 1e10).round(2))
        print("log10 Mstar contribution", np.log10(_mstar_con[:len(prog_idxs)] * 1e10).round(2))
        print("Mstar fractional contribution", (_mstar_con[:len(prog_idxs)] / _mstar_con[0]).round(2))
        print("Major merger?", (_mstar_con[1:len(prog_idxs)] / _mstar_con[0]) > 0.5)
        print("Mstar in-situ", np.log10((imass - np.nansum(_mstar_con[:len(prog_idxs)])) * 1e10).round(2))
        print("Mstar fractional in-situ", ((imass - np.nansum(_mstar_con[:len(prog_idxs)])) / imass).round(2))
        print("===========\n")

# walk the tree back down..
desc_idxs = [prog_idxs[0]]  # set main 'progenitor' to the index of choice initially
for tag_idx in [0, 1, 2, 3, 4]:
    print(fl.tags[tag_idx])
    print("Index,group,subgroup:", desc_idxs[0], group_number[halo][fl.tags[tag_idx]][desc_idxs[0]],
          sgroup_number[halo][fl.tags[tag_idx]][desc_idxs[0]])

    imass = mstar[halo][fl.tags[tag_idx]][desc_idxs[0]]
    # igmass = gmass[fl.tags[tag_idx]][desc_idxs[0]]
    print("log10 Mstar: %.2f" % np.log10(imass * 1e10))
    # print("log10 Mgas: %.2f" % np.log10(igmass * 1e10))
    print("--------")

    output = fl.get_descendants(group_number[halo][fl.tags[tag_idx]][desc_idxs[0]],
                                sgroup_number[halo][fl.tags[tag_idx]][desc_idxs[0]], halo, fl.tags[tag_idx],
                                properties=['desc_group_ids', 'desc_subgroup_ids', 'desc_stellar_masses',
                                            'desc_stellar_mass_contribution', 'desc_gas_masses',
                                            'desc_gas_mass_contribution'])

    grp = output['desc_group_ids']
    sgrp = output['desc_subgroup_ids']
    _mstar = output['desc_stellar_masses']
    _mstar_con = output['desc_stellar_mass_contribution']
    _mgas = output['desc_gas_masses']
    _mgas_con = output['desc_gas_mass_contribution']

    # find indexes in master file
    desc_idxs = [
        np.where((group_number[halo][fl.tags[tag_idx + 1]] == _g) & (sgroup_number[halo][fl.tags[tag_idx + 1]] == _sg))[
            0] for _g, _sg in zip(grp, sgrp)]

    desc_idxs = np.array([_p for _p in desc_idxs if len(_p) > 0]).T[0]

    _mstar_con[_mstar_con == 0.] = np.nan
    _mgas_con[_mgas_con == 0.] = np.nan

    with np.errstate(invalid='ignore'):
        print("Descendants (%s):" % fl.tags[tag_idx + 1])
        print("Index,group,subgroup:", desc_idxs, group_number[halo][fl.tags[tag_idx + 1]][desc_idxs],
              sgroup_number[halo][fl.tags[tag_idx + 1]][desc_idxs])
        print("log10 Mstar:", np.log10(_mstar[:len(desc_idxs)] * 1e10).round(2))
        print("log10 Mstar contribution", np.log10(_mstar_con[:len(desc_idxs)] * 1e10).round(2))
        print("log10 Mgas:", np.log10(_mgas[:len(desc_idxs)] * 1e10).round(2))
        print("log10 Mgas contribution", np.log10(_mgas_con[:len(desc_idxs)] * 1e10).round(2))
        print("===========\n")
