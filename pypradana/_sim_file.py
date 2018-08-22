# Author: Chao Gu, 2018

import re
from os.path import exists, splitext

import numpy as np

__all__ = ['SimFile']


class SimFile():
    """
    Simulation Reader
    -----------------
    Load PRad cooked simulation files (.root), apply event selection cuts and
    save these information into a numpy npz file.

    Parameters
    ----------
    filename : str
        If the file is an rootfile, root_numpy module is used to read the
        data. If the file is a numpy ".npz" file, the arrays are directly
        loaded.
    """

    def __init__(self, filename, *args, **kwargs):
        _, ext = splitext(filename)
        if ext == '.root':
            self.run = int(re.findall(r'\D*_(\d+)\D*\.root', filename)[0])
            self._load_root(filename, *args, **kwargs)
        elif ext == '.npz':
            self.run = int(re.findall(r'\D*_(\d+)\D*\.npz', filename)[0])
            self._load_numpy(filename, *args, **kwargs)
        else:
            raise ValueError('bad filename')

    def _load_root(self, filename, **kwargs):
        import sys
        argv_save = sys.argv
        sys.argv = ['-b', '-n']
        from root_numpy import tree2array
        from ROOT import TFile
        sys.argv = argv_save

        if exists(filename):
            file_ = TFile(filename, 'READ')
            tree = getattr(file_, 'T')
        else:
            return

    def _load_numpy(self, filename):
        loaded = np.load(filename)

    def save(self, filename):
        pass
