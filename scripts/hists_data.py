#!/usr/bin/env python3

import argparse as ap
import pickle
import re
from functools import reduce
from os import listdir
from os.path import join

import numpy as np

from pypradana import DB, Data

parser = ap.ArgumentParser(description='retrive data histograms')
parser.add_argument(
    'production_path', help='path to cooked production data', metavar='path_p')
parser.add_argument(
    'empty_path', help='path to cooked empty data', metavar='path_e')
parser.add_argument(
    '-p',
    '--production',
    nargs=2,
    type=int,
    help='range of production run number',
    metavar=('start_p', 'end_p'),
    dest='range_p',
)
parser.add_argument(
    '-e',
    '--empty',
    nargs=2,
    type=int,
    help='range of empty run number',
    metavar=('start_e', 'end_e'),
    dest='range_e',
)

args = parser.parse_args()

path_p = args.production_path
path_e = args.empty_path
range_p = args.range_p if args.range_p else [0, 10000]
range_e = args.range_e if args.range_e else [0, 10000]

pattern_eb = re.compile(r'([12])gev')
match_eb = re.search(pattern_eb, path_p)
energy = match_eb.group(0)
binning, temp = np.loadtxt(
    'binning_{}.dat'.format(energy),
    usecols=(0, 1),
    unpack=True,
)
bins = np.concatenate((binning, temp[-1:])) / 180 * np.pi


def get_data(path, start, end):
    files = {}
    pattern = re.compile(r'data_(\d+).npz')

    for f in listdir(path):
        match = re.fullmatch(pattern, f)
        if match is not None:
            run = int(match.group(1))
            if start <= run <= end:
                files[run] = join(path, f)

    charge = {}
    result_ep = {}
    result_ee = {}
    for run, file_ in files.items():
        db = DB(run)
        db.ep_e_cut = np.rec.fromrecords(
            [(-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0)],
            dtype=[('min_', np.float32),
                   ('max_', np.float32)],
        )
        data_ep = Data(file_, database=db)

        charge[run] = db.charge

        data_ep.apply_cut(data_ep.get_single_arm_cut('ep'))
        eff_ep = data_ep.get_trigger_efficiency('ep')
        weights_ep = 1 / eff_ep

        hist_ep, _ = np.histogram(data_ep.theta, bins=bins, weights=weights_ep)
        result_ep[run] = hist_ep

        del data_ep

        data_ee = Data(file_, database=db)
        data_ee.apply_cut(data_ee.get_single_arm_cut('ee0'))
        data_ee.apply_cut(data_ee.get_double_arm_cut())
        eff_ee = data_ee.get_trigger_efficiency('ee')
        cut_ee = data_ee.get_single_arm_cut('ee')
        data_ee.apply_cut(cut_ee)
        eff_ee = eff_ee[cut_ee]
        weights_ee = 0.5 / eff_ee

        hist_ee, _ = np.histogram(data_ee.theta, bins=bins, weights=weights_ee)
        result_ee[run] = hist_ee

        del data_ee

    return result_ep, result_ee, charge


ep_p, ee_p, ch_p = get_data(path_p, *range_p)
ep_e, ee_e, ch_e = get_data(path_e, *range_e)

ep_p, ee_p, ch_p, ep_e, ee_e, ch_e = [
    reduce(np.add, x.values()) for x in [ep_p, ee_p, ch_p, ep_e, ee_e, ch_e]
]

dep_p = np.sqrt(ep_p)
dee_p = np.sqrt(ee_p)
dep_e = np.sqrt(ep_e)
dee_e = np.sqrt(ee_e)

ep = ep_p - ep_e * ch_p / ch_e
ee = ee_p - ee_e * ch_p / ch_e
dep = np.sqrt(dep_p**2 + (dep_e * ch_p / ch_e)**2)
dee = np.sqrt(dee_p**2 + (dee_e * ch_p / ch_e)**2)

with open('data_hists_{}.pkl'.format(energy), 'wb') as f:
    save = {
        'ep': ep,
        'ee': ee,
        'dep': dep,
        'dee': dee,
    }
    pickle.dump(save, f)
