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
        if path is None:
            self._path = join(dirname(realpath(__file__)), 'database')
        else:
            self._path = path

        if run in ('1gev', '2gev'):
            self._run = None
            self.beam_type = run
            data_type = 'sim'
        else:
            self._run = run
            if run >= 1362:
                self.beam_type = '2gev'
            else:
                self.beam_type = '1gev'
            data_type = 'data'

        self.hycal_z = 5642.32

        self.e_res = np.array([0.024, 0.062, 0.062], dtype=np.float32)

        self.e_total_cut = 4000 if self.beam_type == '1gev' else 16000
        self.ep_e_cut = np.rec.fromrecords(
            [(-4.0, 4.0), (-4.0, 4.0), (-4.0, 4.0)],
            dtype=[('min_', np.float32),
                   ('max_', np.float32)],
        )
        self.ee_e_cut = np.rec.fromrecords(
            [(-4.0, 4.0), (-4.0, 4.0), (-4.0, 4.0)],
            dtype=[('min_', np.float32),
                   ('max_', np.float32)],
        )
        if self.beam_type == '1gev':
            self.ee2_e_cut = np.rec.fromrecords(
                [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)],
                dtype=[('min_', np.float32),
                       ('max_', np.float32)],
            )
            self.ep_theta_cut = 0.75 / 180 * np.pi
            self.ee_theta_cut = 0.75 / 180 * np.pi
        elif self.beam_type == '2gev':
            self.ee2_e_cut = np.rec.fromrecords(
                [(-6.0, 6.0), (-6.0, 6.0), (-6.0, 6.0)],
                dtype=[('min_', np.float32),
                       ('max_', np.float32)],
            )
            self.ep_theta_cut = 0.7 / 180 * np.pi
            self.ee_theta_cut = 0.7 / 180 * np.pi
        self.ee2_theta0_cut = 0.6 / 180 * np.pi

        self.gem_match_cut = np.array([6.0, 6.0, 6.0], dtype=np.float32)

        self.coplanerity_cut = 10 / 180 * np.pi
        self.vertex_z_cut = 500

        self._load_hycal_modules()

        self._set_dead_modules(
            ['W835', 'W891', 'G775', 'G900', 'G486', 'G732', 'W230'])
        self.dead_module_cut = 1

        if self.beam_type == '1gev':
            virtual_dead_modules = [
                (-38.5, 184.5, 30.0),
                (75.0, 68.5, 15.0),
                (-66.0, -81.0, 15.0),
                (25.0, 72.0, 15.0),
                (81.0, 10.0, 15.0),
            ]
        elif self.beam_type == '2gev':
            virtual_dead_modules = [
                (-19.0, 97.0, 15.0),
            ]
        self.virtual_dead_modules = np.rec.fromrecords(
            virtual_dead_modules,
            dtype=[('x', np.float32),
                   ('y', np.float32),
                   ('r', np.float32)],
        )

        if data_type == 'data':
            self._init_data()
        elif data_type == 'sim':
            self._init_sim()

    def _init_data(self):
        self._load_live_charge()
        self._load_hycal_offsets()
        self._load_s_shape_corrections()
        self._load_trigger_efficiency()
        self._load_gem_efficiency()

    def _init_sim(self):
        self.target_center = -3000 + 89

        self.e_dep_cut = 10
        self.gem_res = 0.1

        Offset = namedtuple('Offset', ['x', 'y'])
        if self.beam_type == '1gev':
            self.e_beam = 1099.65
            self.offset = Offset(0.89712, 1.45704)
        elif self.beam_type == '2gev':
            self.e_beam = 2142
            self.offset = Offset(0.607247, 1.34611)

    def _load_live_charge(self):
        r, charge = np.loadtxt(
            join(self._path, 'beam_charge.dat'),
            dtype=[
                ('r', np.int16),
                ('charge', np.float32),
            ],
            usecols=(0, 3),
            unpack=True,
        )
        self.charge = charge[(r == self._run)][-1]

    def _load_hycal_offsets(self):
        r, name, x, y = np.loadtxt(
            join(self._path, 'coordinates.dat'),
            dtype=[
                ('r', np.int16),
                ('name', np.unicode, 10),
                ('x', np.float32),
                ('y', np.float32),
            ],
            usecols=(0, 1, 2, 3),
            unpack=True,
        )
        selection = (name == 'HyCal') & (r <= self._run)
        Offset = namedtuple('Offset', ['x', 'y'])
        self.offset = Offset(x[selection][-1], y[selection][-1])

    def _load_hycal_modules(self):
        name, type_, size_x, size_y, size_z, x, y, z = np.loadtxt(
            join(self._path, 'hycal_module.txt'),
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

    def _set_dead_modules(self, modules):
        result = []
        for module in modules:
            id_ = (int(module[1:]) + 1000
                   if module.startswith('W') else int(module[1:]))
            result.append(self.modules[(self.modules.cid == id_)])
        self.dead_modules = np.concatenate(result)
        self.dead_modules = self.dead_modules.view(np.recarray)

    def _load_s_shape_corrections(self):
        l0 = np.load(
            join(self._path, 's_shape_ep_{}.npz'.format(self.beam_type)))
        l1 = np.load(
            join(self._path, 's_shape_ee_{}.npz'.format(self.beam_type)))
        SShape = namedtuple('SShape', ['hist', 'edge_x', 'edge_y'])
        self.s_shape = SShape(
            np.stack((l0['hist'], l1['hist'])),
            np.stack((l0['edge_x'], l1['edge_x'])),
            np.stack((l0['edge_y'], l1['edge_y'])),
        )

    def _load_trigger_efficiency(self):
        self.trigger_eff = np.zeros(
            2157,
            dtype=[
                ('p0', np.float32),
                ('p1', np.float32),
                ('p2', np.float32),
            ],
        )
        name, par0, par1, par2 = np.loadtxt(
            join(self._path, 'hycal_trgeff_minTime.txt'),
            dtype=[
                ('name', np.unicode, 10),
                ('par0', np.float32),
                ('par1', np.float32),
                ('par2', np.float32),
            ],
            usecols=(0, 1, 2, 3),
            unpack=True,
        )
        cid = np.char.lstrip(name, 'GW').astype(np.int32)
        cid[np.char.startswith(name, 'W')] += 1000
        it = np.nditer(
            (cid, par0, par1, par2),
            op_flags=[['readonly'], ['readonly'], ['readonly'], ['readonly']],
        )
        for icid, *ipars in it:
            self.trigger_eff[icid] = (ipars[0], ipars[1], ipars[2])
        self.trigger_eff = self.trigger_eff.view(np.recarray)

    def _load_gem_efficiency(self):
        l = np.load(
            join(self._path, 'gem_efficiency_{}.npz'.format(self.beam_type)))
        self.gem_eff = l['hist'].astype(np.float32)
        self.gem_eff_edge = l['edge'].astype(np.float32) / 180 * np.pi

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
        if self.beam_type == '1gev':
            over_flow = (cid == 1194) & (e > 1600) & (e < 1850)
            over_flow |= (cid == 1526) & (e > 1600) & (e < 1900)
            over_flow |= (cid == 1969) & (e > 1550) & (e < 1800)
        elif self.beam_type == '2gev':
            over_flow = (cid == 1194) & (e > 3200) & (e < 3500)
            over_flow |= (cid == 1969) & (e > 3200) & (e < 3500)
        return over_flow
