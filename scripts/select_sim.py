#!/usr/bin/env python3

import argparse as ap
import re
from os import listdir
from os.path import exists, join

from pypradana import SimFile, DB

parser = ap.ArgumentParser(description='select events and save')
parser.add_argument('in_path', help='path to prad simulation files')
parser.add_argument('out_path', help='path to cooked prad simulation files')
parser.add_argument(
    '-e',
    '--energy',
    nargs=1,
    default='1gev',
    help='beam energy',
    dest='energy',
)

args = parser.parse_args()
in_path = args.in_path
out_path = args.out_path
energy = args.energy[0]

for t in ['ep', 'ee', 'in']:
    files = {}
    pattern = re.compile(r'sim_(\d+).npz')
    for f in listdir(join(in_path, t)):
        match = re.fullmatch(pattern, f)
        if match is not None:
            run = int(match.group(1))
            files[run] = join(in_path, t, f)

    db = DB(energy)
    for run, file_ in files.items():
        sim = SimFile(file_, database=db)
        cut_ep = sim.get_single_arm_cut('ep')
        cut_ee = sim.get_single_arm_cut('ee0')
        sim.apply_cut(cut_ep | cut_ee)

        sim.save(join(out_path, t, 'sim_{:04d}.npz'.format(run)))
        del sim
