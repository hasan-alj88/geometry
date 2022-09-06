import datetime
import numpy as np
import subprocess
import sys
from operator import attrgetter

if 'uncertainties' not in sys.modules:
    subprocess.call(['pip', 'install', 'uncertainties'])

from uncertainties import ufloat
from uncertainties import unumpy as unp

from geometry.UfloatOP import UfloatOP
from geometry.Degree import Degree
from geometry.Location import Location


class Sun:
    distance_from_earth: ufloat = UfloatOP.from_minmax(1.47098074 * 10 ** 11, 1.52097701 * 10 ** 11)

    @staticmethod
    def subsolar(utc) -> Location:
        """
        https://medium.com/swlh/is-the-earth-flat-check-for-yourself-2735039b15ea
        :param utc:
        :return:
        """
        ye, mo, da, ho, mi, se = utc
        ut = ho + mi / 60 + se / 3600
        t = 367 * ye - 7 * (ye + (mo + 9) // 12) // 4
        dn = t + 275 * mo // 9 + da - 730531.5 + ut / 24
        sl = dn * 0.01720279239 + 4.894967873
        sa = dn * 0.01720197034 + 6.240040768
        t = sl + 0.03342305518 * np.sin(sa)
        ec = t + 0.0003490658504 * np.sin(2 * sa)
        ob = 0.4090877234 - 0.000000006981317008 * dn
        st = 4.894961213 + 6.300388099 * dn
        ra = np.arctan2(np.cos(ob) * np.sin(ec), np.cos(ec))
        de = np.arcsin(np.sin(ob) * np.sin(ec))
        la = de * 180 / np.pi
        lo = ((ra - st) * 180 / np.pi) % 360
        lo = lo - 360 if lo > 180 else lo
        la = ufloat(round(la, 6), 0.01)
        lo = ufloat(round(lo, 6), 0.01)
        return Location(Latitude=Degree.from_uflloat(la), Longitude=Degree.from_uflloat(lo))

    @property
    def subsolar_now(self) -> Location:
        attrs = ('year', 'month', 'day', 'hour', 'minute', 'second')
        d = datetime.datetime.now()
        d_tuple = attrgetter(*attrs)(d)
        return self.subsolar(d_tuple)