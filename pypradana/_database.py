# Author: Chao Gu, 2018

from collections import namedtuple
from os.path import dirname, join, realpath

import numpy as np

__all__ = ['DB']


class DB():
    """
    PRad Analyzer Database Interface
    --------------------------------
    """

    def __init__(self, run, *, path=None):
        self.hycal_z = 5642.32

        if path is None:
            db_dir = join(dirname(realpath(__file__)), 'database')
        else:
            db_dir = path

        self.e_res = np.array([0.024, 0.062, 0.062])
        self.ep_e_cut = np.rec.fromrecords(
            [(-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0)],
            names='min_,max_',
        )
        self.ee_e_cut = np.rec.fromrecords(
            [(-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0)],
            names='min_,max_',
        )
        self.pos_cut = np.array([6.0, 6.0, 6.0])

        self.dead_modules = [
            'W835', 'W891', 'G775', 'G900', 'G486', 'G732', 'W230'
        ]
        self.dead_module_cut_size = 1

        SShape = namedtuple('SShape', ['hist', 'edge_x', 'edge_y'])
        if run >= 1362:
            self.is1gev = False
            self.is2gev = True

            self.e_total_cut = 16000
            self.ep_theta_cut = 0.7 / 180 * np.pi
            self.ee_theta_cut = 0.7 / 180 * np.pi

            # load s shape correction
            l0 = np.load(join(db_dir, 's_shape_ep_2gev.npz'))
            l1 = np.load(join(db_dir, 's_shape_ee_2gev.npz'))
            self.s_shape = SShape(
                np.stack((l0['hist'], l1['hist'])),
                np.stack((l0['edge_x'], l1['edge_x'])),
                np.stack((l0['edge_y'], l1['edge_y'])),
            )
        else:
            self.is1gev = True
            self.is2gev = False

            self.e_total_cut = 4000
            self.ep_theta_cut = 0.75 / 180 * np.pi
            self.ee_theta_cut = 0.75 / 180 * np.pi

            # load s shape correction
            l0 = np.load(join(db_dir, 's_shape_ep_1gev.npz'))
            l1 = np.load(join(db_dir, 's_shape_ee_1gev.npz'))
            self.s_shape = SShape(
                np.stack((l0['hist'], l1['hist'])),
                np.stack((l0['edge_x'], l1['edge_x'])),
                np.stack((l0['edge_y'], l1['edge_y'])),
            )

        # load live charge
        r, charge = np.loadtxt(
            join(db_dir, 'beam_charge.dat'),
            dtype=[
                ('r', np.int16),
                ('charge', np.float32),
            ],
            usecols=(0, 3),
            unpack=True,
        )
        self.charge = charge[(r == run)][-1]

        # load hycal offsets
        r, name, x, y = np.loadtxt(
            join(db_dir, 'coordinates.dat'),
            dtype=[
                ('r', np.int16),
                ('name', np.unicode, 10),
                ('x', np.float32),
                ('y', np.float32),
            ],
            usecols=(0, 1, 2, 3),
            unpack=True,
        )
        selection = (name == 'HyCal') & (r <= run)
        Offset = namedtuple('Offset', ['x', 'y'])
        self.offset = Offset(x[selection][-1], y[selection][-1])

        # load module list
        name, type_, size_x, size_y, size_z, x, y, z = np.loadtxt(
            join(db_dir, 'hycal_module.txt'),
            dtype=[
                ('name', np.unicode, 10),
                ('type', np.unicode, 10),
                ('size_x', np.float32),
                ('size_y', np.float32),
                ('size_z', np.float32),
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
            ],
            usecols=(0, 1, 2, 3, 4, 5, 6, 7),
            unpack=True,
        )
        cid = np.char.lstrip(name, 'GW').astype(np.int32)
        cid[np.char.startswith(name, 'W')] += 1000
        self.modules = np.rec.fromarrays(
            (cid, type_, size_x, size_y, size_z, x, y, z),
            names='cid,type,size_x,size_y,size_z,x,y,z',
        )

        result = []
        for module in self.dead_modules:
            result.append(self.modules[(name == module)])
        self.dead_modules = np.concatenate(result).view(np.recarray)

    def find_modules(self, x, y):
        from ._tools import _find_modules
        id_ = _find_modules(
            x,
            y,
            self.modules.x,
            self.modules.y,
            self.modules.size_x,
            self.modules.size_y,
        )
        result = self.modules[id_]
        result[(id_ == -1)] = (0, '', 0, 0, 0, 0, 0, 0)
        return result

    def is_over_flow(self, e, cid):
        if self.is1gev:
            over_flow = (cid == 1194) & (e > 1600) & (e < 1850)
            over_flow |= (cid == 1526) & (e > 1600) & (e < 1900)
            over_flow |= (cid == 1969) & (e > 1550) & (e < 1800)
        elif self.is2gev:
            over_flow = (cid == 1194) & (e > 3200) & (e < 3500)
            over_flow |= (cid == 1969) & (e > 3200) & (e < 3500)
        return over_flow
