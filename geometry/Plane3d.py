import datetime
import numpy as np
import pandas as pd
import subprocess
import sys
from dataclasses import dataclass
import re
from operator import attrgetter
from typing import Tuple, List

if 'uncertainties' not in sys.modules:
    subprocess.call(['pip', 'install', 'uncertainties'])

from uncertainties import ufloat
from uncertainties import unumpy as unp
from geometry.Point3d import Point3d
from geometry.Line3d import Line3d


@dataclass
class Plane3d:
    a: ufloat
    b: ufloat
    c: ufloat
    d: ufloat

    @property
    def constants(self):
        return self.a, self.b, self.c, self.d

    def is_on_plane(self, point: Point3d) -> bool:
        a, b, c, d = self.constants
        x, y, z = point.get_cartesian
        return a * x + b * y + c * z + d == 0

    @staticmethod
    def from_3points(p0: Point3d, p1: Point3d, p2: Point3d):
        p0_xyz = list(p0.get_cartesian)
        p0_xyz.append(1)
        p1_xyz = list(p1.get_cartesian)
        p1_xyz.append(1)
        p2_xyz = list(p2.get_cartesian)
        p2_xyz.append(1)
        lhs = np.array([p0_xyz, p1_xyz, p2_xyz])
        a, b, c, d = np.linalg.solve(lhs, np.zeros(3))
        return Plane3d(a=a, b=b, c=c, d=d)

    @property
    def normal(self):
        a, b, c, d = self.constants
        return Point3d(x=a, y=b, z=c).normalized

    def line3d_interception(self, line: Line3d):
        norm = self.normal
        if norm.dot(line.vector) == 0:
            return Point3d.nan()
        a, b, c, d = self.constants
        t = - (norm.dot(line.p0) + d) / norm.dot(line.vector)
        return line.point_at(t=t)

    @staticmethod
    def from_point_and_normal(point: Point3d, normal: Point3d):
        a, b, c = normal.get_cartesian
        d = - point.dot(normal)
        return Plane3d(a=a, b=b, c=c, d=d)
