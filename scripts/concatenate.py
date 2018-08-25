#!/usr/bin/env python3

import argparse as ap
import re
from collections import defaultdict
from os import listdir
from os.path import exists, join

from pypradana import Data, DB

parser = ap.ArgumentParser(description='concatenate data')
parser.add_argument('path', nargs=1, help='path to prad data')

args = parser.parse_args()
path = args.path[0]

files = defaultdict(list)
pattern = re.compile(r'data_(\d+)_(\d+).npz')
for f in listdir(path):
    match = re.fullmatch(pattern, f)
    if match is not None:
        run = int(match.group(1))
        files[run].append(join(path, f))

for run, file_list in files.items():
    if exists(join(path, 'data_{}.npz'.format(run))):
        continue

    if file_list:
        db = DB(run)
        data = Data(file_list[0], database=db)
    for file_ in file_list[1:]:
        more_data = Data(file_, database=db)
        data += more_data
        del more_data

    data.save(join(path, 'data_{}.npz'.format(run)))
    del data
