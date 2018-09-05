#!/usr/bin/env python3

import argparse as ap
from os.path import splitext

import numpy as np

parser = ap.ArgumentParser(description='save gem efficiency as numpy array')
parser.add_argument('file', nargs=1, help='gem efficiency file')

args = parser.parse_args()


def get_hists(filename):
    import sys
    sys.argv.append('-b')
    from ROOT import TFile
    from root_numpy import hist2array

    f = TFile(filename, 'READ')
    h_ep = f.Get('gem_efficiency_ep')
    h_ee = f.Get('gem_efficiency_ee')

    hist_ep, edges = hist2array(h_ep, return_edges=True)
    hist_ee, edges = hist2array(h_ee, return_edges=True)

    return hist_ep[1:], hist_ee[1:], edges[0][1:]


hist_ep, hist_ee, edge = get_hists(args.file[0])

outfile, _ = splitext(args.file[0])
np.savez_compressed(outfile + '.npz', ep=hist_ep, ee=hist_ee, edge=edge)
