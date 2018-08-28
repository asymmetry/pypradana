#!/usr/bin/env python3

import argparse as ap
import re
from os import listdir
from os.path import exists, join

from pypradana import Data, DB

parser = ap.ArgumentParser(
    description='select double arm moller events and save')
parser.add_argument('path', nargs=1, help='path to prad data')

args = parser.parse_args()
path = args.path[0]

files = {}
pattern = re.compile(r'data_(\d+)_ee.npz')
for f in listdir(path):
    match = re.fullmatch(pattern, f)
    if match is not None:
        run = int(match.group(1))
        files[run] = join(path, f)

for run, file_ in files.items():
    if exists(join(path, 'data_{}_ee2.npz'.format(run))):
        continue

    db = DB(run)
    data = Data(file_, database=db)

    cut = data.get_double_arm_cut()
    data.apply_cut(cut)

    data.save(join(path, 'data_{}_ee2.npz'.format(run)))
    del data
