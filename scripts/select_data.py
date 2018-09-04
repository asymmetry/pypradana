#!/usr/bin/env python3

import argparse as ap
import re
from os import listdir
from os.path import exists, join

from pypradana import Data, DB

parser = ap.ArgumentParser(description='select events and save')
parser.add_argument('in_path', help='path to prad data')
parser.add_argument('out_path', help='path to cooked prad data')

args = parser.parse_args()
in_path = args.in_path
out_path = args.out_path

files = {}
pattern = re.compile(r'data_(\d+).npz')
for f in listdir(in_path):
    match = re.fullmatch(pattern, f)
    if match is not None:
        run = int(match.group(1))
        files[run] = join(in_path, f)

for run, file_ in files.items():
    db = DB(run)
    data = Data(file_, database=db)

    cut_ep = data.get_single_arm_cut('ep')
    cut_ee = data.get_single_arm_cut('ee0')
    data.apply_cut(cut_ep | cut_ee)

    data.save(join(out_path, 'data_{}.npz'.format(run)))
    del data
