#! /usr/bin/env python3

import argparse as ap
from os.path import splitext

import numpy as np

parser = ap.ArgumentParser(description='extract the s shape correction data')
parser.add_argument('file', nargs=1, help='s shape correction file')

args = parser.parse_args()

if '_ep_' in args.file[0]:
    type_ = 'ep'
else:
    type_ = 'ee'


def get_hists(filename):
    import sys
    sys.argv.append('-b')
    import ROOT
    from root_numpy import hist2array

    hist_result = np.zeros((2157, 5, 5), dtype=np.float32)
    edge_x_result = np.zeros((2157, 6), dtype=np.float32)
    edge_y_result = np.zeros((2157, 6), dtype=np.float32)
    file_ = ROOT.TFile(filename, 'READ')
    keys = file_.GetListOfKeys()
    for key in keys:
        name = key.GetName()
        id_ = int(name)
        this_dir = key.ReadObj()
        temp_hist = this_dir.Get('avg_{}_ratio_{:04d}'.format(type_, id_))
        hist, edges = hist2array(temp_hist, return_edges=True)
        hist_result[id_] = hist
        edge_x_result[id_] = edges[0]
        edge_y_result[id_] = edges[1]

    return hist_result, edge_x_result, edge_y_result


result = get_hists(args.file[0])

outfile, _ = splitext(args.file[0])
np.savez_compressed(
    outfile + '.npz',
    hist=result[0],
    edge_x=result[1],
    edge_y=result[2],
)
