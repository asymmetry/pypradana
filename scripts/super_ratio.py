#!/usr/bin/env python3

import argparse as ap
import pickle
import re

import numpy as np

from pypradana.tools import get_gem_efficiency

parser = ap.ArgumentParser(description='calculate super ratio')
parser.add_argument('data', help='data result')
parser.add_argument('sim', help='sim result')

args = parser.parse_args()

file_data = args.data
file_sim = args.sim

pattern_eb = re.compile(r'([12])gev')
match_eb = re.search(pattern_eb, file_data)
energy = match_eb.group(0)
binning, temp = np.loadtxt(
    'binning_{}.dat'.format(energy),
    usecols=(0, 1),
    unpack=True,
)
bins = np.concatenate((binning, temp[-1:])) / 180 * np.pi

if energy == '1gev':
    bbb_range = (0.75, 3.5)
    int_range = (1.3, 3.5)
else:
    bbb_range = (0.7, 2.0)
    int_range = (1.3, 2.0)

# load data
with open(file_data, 'rb') as f:
    result_data = pickle.load(f)

with open(file_sim, 'rb') as f:
    result_sim = pickle.load(f)

ep_data = result_data['ep']
ee_data = result_data['ee']
dep_data = result_data['dep']
dee_data = result_data['dee']

ep_sim = result_sim['ep']
ee_sim = result_sim['ee']
dep_sim = result_sim['dep']
dee_sim = result_sim['dee']

theta = (bins[1:] + bins[:-1]) / 2 / np.pi * 180

# get GEM efficiency
gem_eff_ep_data = get_gem_efficiency(theta / 180 * np.pi, energy, 'data', 'ep')
ep_data /= gem_eff_ep_data
dep_data /= gem_eff_ep_data

gem_eff_ee_data = get_gem_efficiency(theta / 180 * np.pi, energy, 'data', 'ee')
ee_data[gem_eff_ee_data > 0] /= gem_eff_ee_data[gem_eff_ee_data > 0]
dee_data[gem_eff_ee_data > 0] /= gem_eff_ee_data[gem_eff_ee_data > 0]

gem_eff_ep_sim = get_gem_efficiency(theta / 180 * np.pi, energy, 'sim', 'ep')
ep_sim /= gem_eff_ep_sim
dep_sim /= gem_eff_ep_sim

gem_eff_ee_sim = get_gem_efficiency(theta / 180 * np.pi, energy, 'sim', 'ee')
ee_sim[gem_eff_ee_sim > 0] /= gem_eff_ee_sim[gem_eff_ee_sim > 0]
dee_sim[gem_eff_ee_sim > 0] /= gem_eff_ee_sim[gem_eff_ee_sim > 0]

# cuts
bbb_cut = (theta < bbb_range[1]) & (theta > bbb_range[0])
int_cut = (theta < int_range[1]) & (theta > int_range[0])

# calculate data ratio
int_ee_data = np.sum(ee_data[int_cut])
dint_ee_data = np.sqrt(np.sum(dee_data[int_cut]**2))

ratio_data = np.zeros_like(ep_data)
ratio_data[bbb_cut] = ep_data[bbb_cut] / ee_data[bbb_cut]
ratio_data[~bbb_cut] = ep_data[~bbb_cut] / int_ee_data

dratio_data = np.zeros_like(dep_data)
dratio_data[bbb_cut] = np.sqrt((dep_data[bbb_cut] / ep_data[bbb_cut])**2 +
                               (dee_data[bbb_cut] / ee_data[bbb_cut])**2)
dratio_data[~bbb_cut] = np.sqrt((dep_data[~bbb_cut] / ep_data[~bbb_cut])**2 +
                                (dint_ee_data / int_ee_data)**2)
dratio_data *= ratio_data

# calculate simulation ratio
int_ee_sim = np.sum(ee_sim[int_cut])
dint_ee_sim = np.sqrt(np.sum(dee_sim[int_cut]**2))

ratio_sim = np.zeros_like(ep_sim)
ratio_sim[bbb_cut] = ep_sim[bbb_cut] / ee_sim[bbb_cut]
ratio_sim[~bbb_cut] = ep_sim[~bbb_cut] / int_ee_sim

dratio_sim = np.zeros_like(dep_sim)
dratio_sim[bbb_cut] = np.sqrt((dep_sim[bbb_cut] / ep_sim[bbb_cut])**2 +
                              (dee_sim[bbb_cut] / ee_sim[bbb_cut])**2)
dratio_sim[~bbb_cut] = np.sqrt((dep_sim[~bbb_cut] / ep_sim[~bbb_cut])**2 +
                               (dint_ee_sim / int_ee_sim)**2)
dratio_sim *= ratio_sim

super_ratio = ratio_sim / ratio_data
dsuper_ratio = np.sqrt((dratio_data / ratio_data)**2 +
                       (dratio_sim / ratio_sim)**2)
dsuper_ratio *= super_ratio

print(super_ratio, dsuper_ratio / super_ratio)

with open('ratio_{}.pkl'.format(energy), 'wb') as f:
    save = {
        'ratio': super_ratio,
        'dratio': dsuper_ratio,
    }
    pickle.dump(save, f)
