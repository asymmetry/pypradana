#!/usr/bin/env python3

import argparse as ap
import re
from os import listdir
from os.path import exists, join

from pypradana import Data, DB

parser = ap.ArgumentParser(description='select ep elastic events and save')
parser.add_argument('path', nargs=1, help='path to prad data')

args = parser.parse_args()
path = args.path[0]

files = {}
pattern = re.compile(r'data_(\d+).npz')
for f in listdir(path):
    match = re.fullmatch(pattern, f)
    if match is not None:
        run = int(match.group(1))
        files[run] = join(path, f)

for run, file_ in files.items():
    if exists(join(path, 'data_{}_ep.npz'.format(run))):
        continue

    db = DB(run)
    data = Data(file_, database=db)

    cut = data.get_single_arm_cut('ep')
    data.apply_cut(cut)

    data.save(join(path, 'data_{}_ep.npz'.format(run)))
    del data
