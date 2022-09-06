import subprocess
import sys
from dataclasses import dataclass

if 'uncertainties' not in sys.modules:
    subprocess.call(['pip', 'install', 'uncertainties'])

from uncertainties import ufloat
from uncertainties import unumpy as unp
from geometry.Point3d import Point3d
from geometry.Location import Location
from geometry.Line3d import Line3d


@dataclass
class Stick:
    location: Location
    length: ufloat

    @property
    def base_point(self) -> Point3d:
        return self.location.point

    @property
    def stick_line(self) -> Line3d:
        return Line3d(p0=Point3d.origin(), p1=self.base_point)
