#!/usr/bin/env python3

import argparse as ap
import pickle
import re
from functools import reduce
from os import listdir
from os.path import join

import numpy as np
from matplotlib import pyplot as plt

from pypradana import DB, Data

parser = ap.ArgumentParser(description='retrive data histograms')
parser.add_argument(
    'production_path', help='path to production data', metavar='path_p')
parser.add_argument('empty_path', help='path to empty data', metavar='path_e')
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
match = re.search(pattern_eb, path_p)
binning, temp = np.loadtxt(
    'binning_{}.dat'.format(match.group(0)),
    usecols=(0, 1),
    unpack=True,
)
bins = np.concatenate((binning, temp[-1:])) / 180 * np.pi


def get_data(path, start, end):
    files_ep = {}
    pattern_ep = re.compile(r'data_(\d+)_ep.npz')
    files_ee = {}
    pattern_ee = re.compile(r'data_(\d+)_ee2.npz')

    for f in listdir(path):
        match_ep = re.fullmatch(pattern_ep, f)
        match_ee = re.fullmatch(pattern_ee, f)
        if match_ep is not None:
            run = int(match_ep.group(1))
            if start <= run <= end:
                files_ep[run] = join(path, f)
        elif match_ee is not None:
            run = int(match_ee.group(1))
            if start <= run <= end:
                files_ee[run] = join(path, f)

    charge = {}
    result_ep = {}
    for run, file_ in files_ep.items():
        db = DB(run)
        db.ep_e_cut = np.rec.fromrecords(
            [(-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0)],
            dtype=[('min_', np.float32),
                   ('max_', np.float32)],
        )
        data = Data(file_, database=db)

        charge[run] = db.charge

        eff_t = data.get_trigger_efficiency('ep')
        weights = 1 / eff_t

        hist, _ = np.histogram(data.theta, bins=bins, weights=weights)
        result_ep[run] = hist

        del data

    result_ee = {}
    for run, file_ in files_ee.items():
        db = DB(run)
        db.ep_e_cut = np.rec.fromrecords(
            [(-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0)],
            dtype=[('min_', np.float32),
                   ('max_', np.float32)],
        )
        data = Data(file_, database=db)

        eff_t = data.get_trigger_efficiency('ee')
        weights = 0.5 / eff_t

        hist, _ = np.histogram(data.theta, bins=bins, weights=weights)
        result_ee[run] = hist

        del data

    return result_ep, result_ee, charge


ep_p, ee_p, ch_p = get_data(path_p, *range_p)
ep_e, ee_e, ch_e = get_data(path_e, *range_e)

with open('data_hists.pkl', 'wb') as f:
    save = {
        'ep_p': ep_p,
        'ee_p': ee_p,
        'ch_p': ch_p,
        'ep_e': ep_e,
        'ee_e': ee_e,
        'ch_e': ch_e,
    }
    pickle.dump(save, f)
