#!/usr/bin/env python3

import argparse as ap
import re
from os import listdir
from os.path import exists, isfile, join, splitext

from pypradana import Data

parser = ap.ArgumentParser(description='extract data')
parser.add_argument('path', nargs=1, help='path to prad data')

args = parser.parse_args()
path = args.path[0]

files = [
    join(path, f) for f in listdir(path)
    if isfile(join(path, f)) and splitext(f)[1] == '.root'
]
print(files)


def get_n_entries(file_):
    import sys
    sys.argv.append('-b')
    from ROOT import TFile

    f = TFile(file_, 'READ')
    tree = getattr(f, 'T')

    n_entries = tree.GetEntries()
    del TFile
    return n_entries


for file_ in files:
    run = int(re.findall(r'\D*_(\d+)\D*\.root', file_)[0])

    n = get_n_entries(file_) // 1000000

    for i in range(n + 1):
        if exists(join(path, 'data_{}_{}.npz'.format(run, i + 1))):
            continue

        try:
            data = Data(file_, start=i * 1000000, stop=(i + 1) * 1000000)
            data.module_e_correction()
        except (AttributeError, ValueError):
            continue

        data.save(join(path, 'data_{}_{}.npz'.format(run, i + 1)))
        del data
