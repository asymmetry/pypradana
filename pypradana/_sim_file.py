# Author: Chao Gu, 2018

from os.path import exists, splitext

import numpy as np

from ._database import DB

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

    def __init__(self, filename, *, database=None, **kwargs):
        if database is None:
            self.db = DB(1415)
        else:
            self.db = database

        _, ext = splitext(filename)
        if ext == '.root':
            self._load_root(filename, **kwargs)
        elif ext == '.npz':
            self._load_numpy(filename)
        else:
            raise ValueError('bad filename')

    def __add__(self, other):
        for key, value in vars(self).items():
            if isinstance(value, np.ndarray):
                setattr(
                    self,
                    key,
                    np.concatenate((value, getattr(other, key))),
                )
        return self

    def _load_root(self, filename, **kwargs):
        print('loading {} ...... '.format(filename), end='', flush=True)

        import sys
        sys.argv.append('-b')
        from ROOT import TFile
        from root_numpy import tree2array

        if exists(filename):
            file_ = TFile(filename, 'READ')
            tree1 = getattr(file_, 'T')
            tree2 = tree1
        else:
            basename, ext = splitext(filename)
            if (exists(basename + '_rec' + ext)
                    and exists(basename + '_red' + ext)):
                file1 = TFile(basename + '_rec' + ext, 'READ')
                file2 = TFile(basename + '_red' + ext, 'READ')
                tree1 = getattr(file1, 'T')
                tree2 = getattr(file2, 'T')
            else:
                raise FileNotFoundError('{} does not exists!'.format(filename))

        branches = [
            'EventNumber',
            'EBeam',
            'TotalE',
            ('Hit.Flag', 0, 100),
            ('Hit.Match', 0, 100),
            ('Hit.X', 0, 100),
            ('Hit.Y', 0, 100),
            ('Hit.Z', 0, 100),
            ('Hit.X1', 0, 100),
            ('Hit.Y1', 0, 100),
            ('Hit.Z1', 0, 100),
            ('Hit.E', 0, 100),
            ('Hit.NModule', 0, 100),
            ('Hit.CID', 0, 100),
            ('Hit.GEM.X', 0, 100),
            ('Hit.GEM.Y', 0, 100),
            ('Hit.GEM.Z', 0, 100),
        ]

    def _load_numpy(self, filename):
        loaded = np.load(filename)

    def save(self, filename):
        pass
