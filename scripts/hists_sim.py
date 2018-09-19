#!/usr/bin/env python3

import argparse as ap
import pickle
import re
from functools import reduce
from os import listdir
from os.path import join

import numpy as np

from pypradana import DB, SimFile

parser = ap.ArgumentParser(description='retrive simulation histograms')
parser.add_argument('path', help='path to simulations')

args = parser.parse_args()
path = args.path

pattern_eb = re.compile(r'([12])gev')
match_eb = re.search(pattern_eb, path)
energy = match_eb.group(0)
binning, temp = np.loadtxt(
    'binning_{}.dat'.format(energy),
    usecols=(0, 1),
    unpack=True,
)
bins = np.concatenate((binning, temp[-1:])) / 180 * np.pi


def get_data(t):
    files = {}
    pattern = re.compile(r'sim_(\d+).npz')
    for f in listdir(join(path, t)):
        match = re.fullmatch(pattern, f)
        if match is not None:
            run = int(match.group(1))
            files[run] = join(path, t, f)

    result_ep = {}
    result_ee = {}
    db = DB(energy)
    db.ep_e_cut = np.rec.fromrecords(
        [(-4.0, 4.0), (-2.0, 4.0), (-2.0, 4.0)],
        dtype=[('min_', np.float32),
               ('max_', np.float32)],
    )
    # db.ee_e_cut = np.rec.fromrecords(
    #     [(-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0)],
    #     dtype=[('min_', np.float32),
    #            ('max_', np.float32)],
    # )
    for run, file_ in files.items():
        sim_ep = SimFile(file_, database=db)

        sim_ep.apply_cut(sim_ep.get_single_arm_cut('ep'))

        hist_ep, _ = np.histogram(sim_ep.theta, bins=bins)
        result_ep[run] = hist_ep

        del sim_ep

        sim_ee = SimFile(file_, database=db)
        sim_ee.apply_cut(sim_ee.get_single_arm_cut('ee0'))
        sim_ee.apply_cut(sim_ee.get_double_arm_cut())
        sim_ee.apply_cut(sim_ee.get_single_arm_cut('ee'))

        hist_ee, _ = np.histogram(sim_ee.theta, bins=bins)
        result_ee[run] = hist_ee

        del sim_ee

    return result_ep, result_ee


ep_ep, ee_ep = get_data('ep')
ep_ee, ee_ee = get_data('ee')
ep_in, ee_in = get_data('in')

ep_ep, ee_ep, ep_ee, ee_ee, ep_in, ee_in = [
    reduce(np.add, x.values())
    for x in [ep_ep, ee_ep, ep_ee, ee_ee, ep_in, ee_in]
]

# 1 gev
lumi_ep = 0.789009635
lumi_ee = 0.5
lumi_in = 0.020668338 / 2 * 3
# 2 gev
lumi_ep = 0.903422618
lumi_ee = 0.5
lumi_in = 0.05636695

ep = ep_ep * lumi_ep + ep_ee * lumi_ee + ep_in * lumi_in
ee = ee_ep * lumi_ep + ee_ee * lumi_ee + ee_in * lumi_in
dep = np.sqrt(ep)
dee = np.sqrt(ee)

theta = (bins[1:] + bins[:-1]) / 2 / np.pi * 180

with open('sim_hists_{}.pkl'.format(energy), 'wb') as f:
    save = {
        'ep': ep,
        'ee': ee,
        'dep': dep,
        'dee': dee,
    }
    pickle.dump(save, f)
