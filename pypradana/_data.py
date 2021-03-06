# Author: Chao Gu, 2018

import re
from functools import reduce
from os.path import exists, splitext

import numpy as np

from ._database import DB
from .tools import get_elastic_energy

__all__ = ['Data']


class Data():
    """
    Data Reader
    -----------
    Load PRad cooked loaded files (.root), apply event selection cuts and save
    these information into a numpy npz file.

    Parameters
    ----------
    filename : str
        If the file is an rootfile, root_numpy module is used to read the
        loaded. If the file is a numpy ".npz" file, the loaded are directly
        loaded.
    """

    def __init__(self, filename, *, database=None, **kwargs):
        possible_run = re.findall(r'\D*_(\d+)\D*\.\D*', filename)
        if possible_run:
            self.run = int(possible_run[0])

        if database is None:
            self.db = DB(self.run, path=kwargs.get('path', None))
        else:
            self.db = database

        _, ext = splitext(filename)
        if ext == '.root':
            self._load_root(filename, **kwargs)
        elif ext == '.npz':
            self._load_numpy(filename)
        else:
            raise FileNotFoundError('{} does not exists!'.format(filename))

    def __add__(self, other):
        for key, value in vars(self).items():
            if isinstance(value, np.ndarray):
                setattr(
                    self,
                    key,
                    np.concatenate((value, getattr(other, key))),
                )
        return self

    def _load_root(self, filename, *, start=None, stop=None, **_):
        print('loading {} ...... '.format(filename), end='', flush=True)

        import sys
        sys.argv.append('-b')
        from ROOT import TFile
        from root_numpy import tree2array

        if exists(filename):
            file_ = TFile(filename, 'READ')
            tree = getattr(file_, 'T')
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
        loaded = tree2array(tree, branches=branches, start=start, stop=stop)

        # only select valid hits
        flag = loaded['Hit.Flag']
        mask = flag > 0

        self.event_number = loaded['EventNumber']
        self.e_beam = loaded['EBeam']
        self.e_total = loaded['TotalE']

        # broadcast event_number, e_beam, e_total to the same shape of hit
        b_shape = flag.shape
        broadcast = lambda x: np.broadcast_to(x[:, np.newaxis], b_shape)
        self.event_number = (broadcast(self.event_number))[mask]
        self.e_beam = (broadcast(self.e_beam))[mask]
        self.e_total = (broadcast(self.e_total))[mask]

        self.e = loaded['Hit.E'][mask]
        self.x = loaded['Hit.GEM.X'][mask]
        self.y = loaded['Hit.GEM.Y'][mask]
        self.z = loaded['Hit.GEM.Z'][mask]
        self.x0 = loaded['Hit.X'][mask]
        self.y0 = loaded['Hit.Y'][mask]
        self.z0 = loaded['Hit.Z'][mask]
        self.x1 = loaded['Hit.X1'][mask]
        self.y1 = loaded['Hit.Y1'][mask]
        self.z1 = loaded['Hit.Z1'][mask]
        self.n_module = loaded['Hit.NModule'][mask]
        self.id_module = loaded['Hit.CID'][mask]

        # hit flag
        flag = flag[mask]
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

        # GEM match flag
        match = loaded['Hit.Match'][mask]
        matches = {}
        for i in range(8, 10):
            matches[i] = (np.bitwise_and(match, int('1' + '0' * i, 2)) != 0)
        self._gem_match = matches[8] | matches[9]  # kGEM1Match & kGEM2Match

        # calculate angles
        self.theta = np.arctan(
            np.sqrt(self.x**2 + self.y**2) / self.db.hycal_z)
        self.phi = np.arctan2(self.y, self.x)

        self.theta0 = np.arctan(
            np.sqrt(self.x0**2 + self.y0**2) / self.db.hycal_z)
        self.phi0 = np.arctan2(self.y0, self.x0)

        print('done!')

    def _load_numpy(self, filename):
        print('loading {} ...... '.format(filename), end='', flush=True)

        loaded = np.load(filename)
        for key, value in loaded.items():
            setattr(self, key, value)

        print('done!')

    @property
    def q2(self, mode='ep'):
        if mode == 'ep':
            e_elastic = get_elastic_energy(self.e_beam, self.theta, 'proton')
            result = 4 * self.e_beam * e_elastic * (np.sin(self.theta / 2))**2
        elif mode == 'ee':
            e_elastic = get_elastic_energy(self.e_beam, self.theta, 'electron')
            result = 4 * self.e_beam * e_elastic * (np.sin(self.theta / 2))**2
        return result

    def save(self, filename):
        print('saving data to {} ...... '.format(filename), end='', flush=True)

        array_dict = {}
        for key, value in vars(self).items():
            if isinstance(value, np.ndarray):
                array_dict[key] = value
        np.savez_compressed(filename, **array_dict)

        print('done!')

    def apply_cut(self, cut):
        for key, value in vars(self).items():
            if isinstance(value, np.ndarray):
                setattr(self, key, value[cut])

    def correct_module_e(self):
        from ._tools import _get_module_e_correction

        e_elastic = get_elastic_energy(self.e_beam, self.theta0, 'proton')

        # decide correction type
        correct_type = np.zeros_like(self.theta0, dtype=np.int32)

        e_cut = self.db.e_res[self.region]
        e_cut *= e_elastic / np.sqrt(e_elastic / 1000)

        cut_size = self.db.ep_e_cut.min_[self.region] - 2.5
        cut_size[self.region != 0] -= 1.5

        t_cut = 2.5 / 180 * np.pi
        e_cut = e_elastic + cut_size * e_cut
        correct_type[(self.theta0 < t_cut) & (self.e <= e_cut)] = 1
        if self.db.beam_type == '1gev':
            correct_type[(self.theta0 >= t_cut) & (self.e < 600)] = 1
        elif self.db.beam_type == '2gev':
            correct_type[(self.theta0 >= t_cut) & (self.e < 1500)] = 1

        # find module and calculate correction
        factor = np.zeros_like(self.e, dtype=np.float)
        modules = self.db.find_modules(self.x1, self.y1)
        factor[(modules.cid == 0)] = 1

        s = (modules.cid > 0)
        factor[s] = _get_module_e_correction(
            (self.x1[s] - modules.x[s]) / modules.size_x[s],
            (self.y1[s] - modules.y[s]) / modules.size_y[s],
            modules.cid[s],
            correct_type[s],
            self.db.s_shape.hist,
            self.db.s_shape.edge_x,
            self.db.s_shape.edge_y,
        )

        self.e /= factor

    @property
    def common_cut(self):
        from ._tools import _not_at_dead_module

        cuts = []

        # e total cut
        cuts.append(self.e_total < self.db.e_total_cut)

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

    @property
    def single_arm_common_cut(self):
        from ._tools import _not_at_dead_module

        cuts = []

        # cluster size cut
        cuts.append(self.n_module > 2)

        # only PbWO4 cut
        # cuts.append((np.abs(self.x) < 342.705) & (np.abs(self.x) < 342.375))

        # GEM strip cut (not implemented)
        # cuts.append()

        # module size cut (not implemented)
        # cuts.append()

        # virtual dead module cut (use GEM position)
        cut0 = _not_at_dead_module(
            self.x,
            self.y,
            self.db.virtual_dead_modules.x,
            self.db.virtual_dead_modules.y,
            self.db.virtual_dead_modules.r,
        )
        cuts.append(cut0.astype(np.bool))

        return reduce(np.bitwise_and, cuts)

    @property
    def gem_match(self):
        cut0 = self._gem_match
        cut1 = (np.abs(self.x) <= 520) & (np.abs(self.y) <= 520)
        delta = np.sqrt((self.x - self.x0)**2 + (self.y - self.y0)**2)
        cut2 = np.zeros_like(cut0, dtype=np.bool)
        e = self.e / 1000
        se = np.sqrt(e)
        delta_0 = self.db.gem_match_cut[0]
        delta_0 *= 2.44436 / se + 0.109709 / e - 0.0176315
        delta_1 = self.db.gem_match_cut[1] * (6.5 / se)
        delta_2 = self.db.gem_match_cut[2] * (6.5 / se)
        cut2[self.at_pwo] = (delta[self.at_pwo] < delta_0[self.at_pwo])
        cut2[self.at_glass] = (delta[self.at_glass] < delta_2[self.at_glass])
        cut2[self.at_trans] = (delta[self.at_trans] < delta_1[self.at_trans])

        return cut0 & cut1 & cut2

    def get_single_arm_cut(self, mode='ep'):
        cuts = []
        cuts.append(self.common_cut)

        if mode == 'ep':
            cuts.append(self.single_arm_common_cut)
            cuts.append(self.gem_match)
            theta = self.theta
            theta_cut = self.db.ep_theta_cut
            e_cut = self.db.ep_e_cut
            recoil = 'proton'
        elif mode == 'ee':
            cuts.append(self.single_arm_common_cut)
            cuts.append(self.gem_match)
            theta = self.theta
            theta_cut = self.db.ee_theta_cut
            e_cut = self.db.ee_e_cut
            recoil = 'electron'
        elif 'ee' in mode and ('hycal' in mode or '0' in mode):
            theta = self.theta0
            theta_cut = self.db.ee2_theta0_cut
            e_cut = self.db.ee2_e_cut
            recoil = 'electron'
        else:
            raise ValueError('mode must be gem / hycal')

        # theta cut
        cuts.append((theta > theta_cut))

        # phi cut (not implemented)
        # cuts.append()

        # energy cut
        e_elastic = get_elastic_energy(self.e_beam, theta, recoil)
        e_res = self.db.e_res[self.region]
        e_res *= e_elastic / np.sqrt(e_elastic / 1000)
        deviation = self.e - e_elastic
        cut0 = (deviation > (e_cut.min_[self.region] * e_res))
        cut1 = (deviation < (e_cut.max_[self.region] * e_res))
        cuts.append(cut0 & cut1)

        return reduce(np.bitwise_and, cuts)

    def get_double_arm_cut(self):
        from ._tools import _get_double_arm_cut

        e_elastic = get_elastic_energy(self.e_beam, self.theta0, 'electron')
        e_res = self.db.e_res[self.region]
        e_res = e_res * e_elastic / np.sqrt(e_elastic / 1000)
        e_cut_min = self.db.ee_e_cut.min_[self.region] * e_res
        e_cut_max = self.db.ee_e_cut.max_[self.region] * e_res

        r0 = np.sqrt(self.x0**2 + self.y0**2)

        cut = _get_double_arm_cut(
            self.event_number,
            self.e_beam,
            self.e,
            e_cut_min,
            e_cut_max,
            self.phi0,
            r0,
            np.float32(self.db.coplanerity_cut),
            np.float32(self.db.hycal_z),
            np.float32(self.db.vertex_z_cut),
        )
        return cut.astype(np.bool)

    def get_trigger_efficiency(self, mode='ep'):
        eff = self.db.trigger_eff[self.id_module]

        if mode == 'ep':
            result = eff.p0 * (1 - np.exp(-eff.p1 * (self.e / 1000) - eff.p2))
            result[(result < 0) | (result > 1)] = 1
        elif mode == 'ee':
            e_sum = np.zeros_like(self.e, dtype=np.float32)
            e_sum[::2] = self.e[::2] + self.e[1::2]
            e_sum[1::2] = e_sum[::2]
            temp = eff.p0 * (1 - np.exp(-eff.p1 * (e_sum / 1000) - eff.p2))
            result = np.zeros_like(temp, dtype=np.float32)
            result[::2] = (temp[::2] * self.e[::2] +
                           temp[1::2] * self.e[1::2]) / e_sum[::2]
            result[1::2] = result[::2]

        return result
