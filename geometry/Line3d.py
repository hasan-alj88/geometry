from geometry.UfloatOP import UfloatOP
from geometry.Point3d import Point3d
from uncertainties import ufloat
import numpy as np
from dataclasses import dataclass


@dataclass
class Line3d:
    p0: Point3d
    p1: Point3d

    @property
    def a(self):
        return self.p1.x - self.p0.x

    @property
    def b(self):
        return self.p1.y - self.p0.y

    @property
    def c(self):
        return self.p1.z - self.p0.z

    @property
    def v(self):
        return UfloatOP.hypotenuse(self.a, self.b, self.c)

    @property
    def vector(self):
        a, b, c, x0, y0, z0 = self.constants
        return Point3d(x=a, y=b, z=c)

    @property
    def constants(self):
        return self.a, self.b, self.c, self.p0.x, self.p0.y, self.p0.z

    def is_on_line(self, point: Point3d) -> bool:
        a, b, c, x0, y0, z0 = self.constants
        tx = (point.x - x0) / a
        ty = (point.y - y0) / b
        tz = (point.z - z0) / c
        return tx == ty == tz

    def point_at(self, t: ufloat) -> Point3d:
        a, b, c, x0, y0, z0 = self.constants
        return Point3d(
            x=x0 + a * t,
            y=y0 + b * t,
            z=z0 + c * t
        )

    def after_by_length(self, point: Point3d, length: ufloat) -> Point3d:
        if not self.is_on_line(point):
            return Point3d.nan()
        dt = length / self.v
        t = (point.x - self.p0.x) / self.a
        return self.point_at(t + dt)

    @classmethod
    def from_point_and_vector(cls, point: Point3d, vector: Point3d):
        return cls(
            p0=point,
            p1=point + vector.normalized
        )

    def interception_line(self, line):
        a0, b0, c0, x0, y0, z0 = self.constants
        a1, b1, c1, x1, y1, z1 = line.constants
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0

        txy1, txy2 = np.linalg.solve(a=[[a0, -a1], [b0, -b1]], b=[dx, dy])
        tyz1, tyz2 = np.linalg.solve(a=[[b0, -b1], [c0, -c1]], b=[dy, dz])
        if txy1 == tyz1 and txy1 == txy2:
            return self.point_at(txy1)
        else:
            return Point3d.nan()
