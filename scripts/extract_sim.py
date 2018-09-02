#!/usr/bin/env python3

import argparse as ap
import re
from os import listdir
from os.path import exists, join

from pypradana import DB, SimFile

parser = ap.ArgumentParser(description='extract simulations')
parser.add_argument('path', nargs=1, help='path to prad simulation files')
parser.add_argument(
    '-e',
    '--energy',
    nargs=1,
    default='1gev',
    help='beam energy',
    dest='energy',
)

args = parser.parse_args()
path = args.path[0]
energy = args.energy[0]

files = {}
pattern = re.compile(r'(.*)_(\d+)_e-_rec.root')
for f in listdir(path):
    match = re.fullmatch(pattern, f)
    if match is not None:
        name = match.group(1)
        run = int(match.group(2))
        files[run] = join(path, '{}_{}_e-.root'.format(name, run))

for run, file_ in files.items():
    if exists(join(path, 'sim_{:04d}.npz'.format(run))):
        continue

    try:
        db = DB(energy)
        data = SimFile(file_, database=db)
    except (AttributeError, ValueError):
        continue

    data.save(join(path, 'sim_{:04d}.npz'.format(run)))
    del data
