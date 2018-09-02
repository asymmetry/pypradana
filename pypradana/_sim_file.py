# Author: Chao Gu, 2018

from functools import reduce
from os.path import exists, splitext

import numpy as np

from ._data import Data
from ._database import DB

__all__ = ['SimFile']


class SimFile(Data):
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

    def _load_root(self, filename, *, start=None, stop=None, **_):
        print('loading {} ...... '.format(filename), end='', flush=True)

        import sys
        sys.argv.append('-b')
        from ROOT import TFile
        from root_numpy import tree2array

        from ._tools import _is_inside_gem_spacers, _match_hits

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

        branches1 = [
            ('HC.X', 0, 100),
            ('HC.Y', 0, 100),
            ('HC.Z', 0, 100),
            ('HC.P', 0, 100),
            ('HC.CID', 0, 100),
            ('HC.Flag', 0, 100),
            ('HC.NModule', 0, 100),
        ]
        loaded1 = tree2array(tree1, branches=branches1, start=start, stop=stop)
        branches2 = [
            ('GEM.X', 0, 100),
            ('GEM.Y', 0, 100),
            ('GEM.Z', 0, 100),
            ('GEM.DID', -1, 100),
            'HC.TotalEdep',
        ]
        loaded2 = tree2array(tree2, branches=branches2, start=start, stop=stop)

        self.e = loaded1['HC.P']
        n_entries = len(self.e)

        # sort the HyCal variables with energy
        b_shape = self.e.shape
        broadcast = lambda x: np.broadcast_to(x[:, np.newaxis], b_shape)
        rows = broadcast(np.arange(n_entries))
        cols = np.argsort(self.e, axis=1)[:, ::-1]
        self.e = self.e[rows, cols]

        mask1 = self.e > 0
        mask2 = loaded2['GEM.DID'] > -1
        indices = (rows[mask1], cols[mask1])

        self.x = loaded2['GEM.X'][mask2]
        self.y = loaded2['GEM.Y'][mask2]
        self.z = loaded2['GEM.Z'][mask2]
        self.x0 = loaded1['HC.X'][indices]
        self.y0 = loaded1['HC.Y'][indices]
        self.z0 = loaded1['HC.Z'][indices]
        self.n_module = loaded1['HC.NModule'][indices]
        self.id_module = loaded1['HC.CID'][indices]

        self.event_number = broadcast(np.arange(1, n_entries + 1))[indices]
        self.e_dep = broadcast(loaded2['HC.TotalEdep'])[indices]

        self.e = self.e[mask1]

        # add the offsets to coordinates
        self.x0 += self.db.offset.x
        self.y0 += self.db.offset.y
        self.x += self.db.offset.x
        self.y += self.db.offset.y

        # hit flag
        flag = loaded1['HC.Flag'][indices]
        flags = {}
        for i in range(9):
            flags[i] = (np.bitwise_and(flag, int('1' + '0' * i, 2)) != 0)
        self.at_pwo = flags[1]  # kPbWO4
        self.at_trans = flags[2]  # kTransition
        self.at_glass = flags[0]  # kPbGlass
        self.region = np.zeros_like(flag, dtype=np.int16)
        self.region[self.at_trans] = 1
        self.region[self.at_glass] = 2
        self.outer_bound = flags[7]  # kOuterBound

        # smear GEM variables
        self.x *= -1
        self.x += np.random.normal(scale=self.db.gem_res, size=self.x.size)
        self.y += np.random.normal(scale=self.db.gem_res, size=self.x.size)

        # project HyCal coordinates
        factor = self.db.hycal_z / (self.z0 - self.db.target_center)
        self.x0 *= factor
        self.y0 *= factor

        # project GEM coordinates
        factor = self.db.hycal_z / (self.z - self.db.target_center)
        self.x *= factor
        self.y *= factor

        # remove GEM spacers
        cut = _is_inside_gem_spacers(self.x, self.y)
        cut = cut.astype(np.bool)
        self.x[cut] = -9999
        self.y[cut] = -9999

        # match GEM with HyCal
        cut_size = self.db.gem_match_cut[self.region]
        e = self.e / 1000
        se = np.sqrt(e)
        pos_res = np.where(
            self.region == 0,
            2.44436 / se + 0.109709 / e - 0.0176315,
            6.5 / se,
        )
        cut = cut_size * pos_res
        gem_event_number = broadcast(np.arange(1, n_entries + 1))[mask2]
        indices = _match_hits(
            self.event_number,
            self.x0,
            self.y0,
            gem_event_number,
            self.x,
            self.y,
            cut,
        )
        self._gem_match = (indices >= 0)
        self.x = np.where(self._gem_match, self.x[indices], 0)
        self.y = np.where(self._gem_match, self.y[indices], 0)

        self.e_beam = np.full_like(self.e, self.db.e_beam)

        # calculate angles
        self.theta = np.arctan(
            np.sqrt(self.x**2 + self.y**2) / self.db.hycal_z)
        self.phi = np.arctan2(self.y, self.x)

        self.theta0 = np.arctan(
            np.sqrt(self.x0**2 + self.y0**2) / self.db.hycal_z)
        self.phi0 = np.arctan2(self.y0, self.x0)

        print('done!')

    @property
    def common_cut(self):
        from ._tools import _not_at_dead_module

        cuts = []

        # e dep cut
        cuts.append(self.e_dep > self.db.e_dep_cut)

        # outer boundary cut
        cuts.append(~(self.outer_bound))

        # dead module cut (use HyCal position)
        cut_size = self.db.dead_module_cut
        cut_size *= (
            self.db.dead_modules.size_x + self.db.dead_modules.size_y) / 2
        module_x = self.db.dead_modules.x + self.db.offset.x
        module_y = self.db.dead_modules.y + self.db.offset.y

        cut0 = _not_at_dead_module(
            self.x0,
            self.y0,
            module_x,
            module_y,
            cut_size,
        )
        cuts.append(cut0.astype(np.bool))

        return reduce(np.bitwise_and, cuts)
