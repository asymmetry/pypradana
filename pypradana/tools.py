# Author: Chao Gu, 2018

import numpy as np
from scipy import constants

_m_e = constants.value('electron mass energy equivalent in MeV')
_m_p = constants.value('proton mass energy equivalent in MeV')
_m_e2 = _m_e**2
_m_p2 = _m_p**2


def get_elastic_energy(e, theta, select):
    s = np.sin(theta)
    c = np.cos(theta)
    if select == 'proton':
        result = (e + _m_p) * (_m_p * e + _m_e2)
        result += np.sqrt(_m_p2 - (_m_e * s)**2) * (e**2 - _m_e2) * c
        result /= (e + _m_p)**2 - (e**2 - _m_e2) * c**2
    elif select == 'electron':
        result = _m_e * (e + _m_e + (e - _m_e) * c**2)
        result /= e + _m_e - (e - _m_e) * c**2
    return result
