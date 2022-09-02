from geometry.UfloatOP import UfloatOP
from geometry.Line3d import Line3d
from geometry.Point3d import Point3d
from uncertainties import ufloat
from uncertainties import unumpy as unp
import numpy as np
from dataclasses import dataclass


@dataclass
class Sphere:
    radius: ufloat
    center: Point3d = Point3d.origin()

    def intercept_line3d(self, line: Line3d):
        a, b, c, x0, y0, z0 = line.constants
        dx, dy, dz = x0 - self.center.x, y0 - self.center.y, z0 - self.center.z

        A = (a ** 2 + b ** 2 + c ** 2)
        B = 2 * (a + b + c)
        C = (dx ** 2 + dy ** 2 + dz ** 2 - self.radius ** 2)

        det = B ** 2 - 4 * A * C
        if det < 0:
            return Point3d.nan(), Point3d.nan()

        t1 = (-B + unp.sqrt(det)) / (2 * A)
        t2 = (-B - unp.sqrt(det)) / (2 * A)

        return line.point_at(t1), line.point_at(t2)

    def is_on_sphere(self, point: Point3d) -> bool:
        return (self.center - point).r == self.radius

    def arc_distance(self, p0: Point3d, p1: Point3d) -> ufloat:
        if not self.is_on_sphere(p0):
            raise ValueError(f'Point {p0} is not on the sphere {self}')
        if not self.is_on_sphere(p1):
            raise ValueError(f'Point {p1} is not on the sphere {self}')

        p2 = p0 - self.center
        p3 = p1 - self.center
        ang = p2.angle(p3)

        return ang * self.radius

    @property
    def great_circle_circumference(self):
        return 2 * np.pi * self.radius

    def small_circle_circumference(self, angle: ufloat):
        return 2 * np.pi * self.radius * unp.sin(angle)
