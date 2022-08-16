import numpy as np
import pandas as pd
from dataclasses import dataclass
import sys
import subprocess
import datetime
from operator import attrgetter

if 'uncertainties' not in sys.modules:
    subprocess.call(['pip', 'install', 'uncertainties'])

from uncertainties import ufloat
from uncertainties import umath


def hpot(x: ufloat, y: ufloat, z: ufloat):
    return umath.sqrt(z ** 2 + y ** 2 + z ** 2)


def uncertainty_between(maxval: float, minval: float):
    return ufloat((maxval + minval) / 2, abs(maxval - minval) / 2)


@dataclass
class Point3d:
    x: ufloat
    y: ufloat
    z: ufloat

    @property
    def r(self):
        return hpot(self.x, self.y, self.z)

    @property
    def theta(self) -> ufloat:
        return umath.cos(self.z / self.r)

    @property
    def phi(self):
        if self.x > 0:
            return umath.arctan(self.y / self.x)
        elif self.x < 0 <= self.y:
            return umath.arctan(self.y / self.x) + np.pi
        elif self.x < 0 and self.y < 0:
            return umath.arctan(self.y / self.x) - np.pi
        elif self.x == 0 and self.y > 0:
            return ufloat(np.pi / 2, self.x.std_dev)
        elif self.x == 0 and self.y < 0:
            return ufloat(-np.pi / 2, self.x.std_dev)
        else:
            return ufloat(np.nan, 0)

    @property
    def get_cartesian(self):
        return self.x, self.y, self.z

    @property
    def get_spherical(self):
        return self.r, self.theta, self.phi

    def __add__(self, other):
        x0, y0, z0 = self.get_cartesian
        x1, y1, z1 = other.get_cartesian
        return Point3d(x=x0 + x1, y=y0 + y1, z=z0 + z0)

    def __sub__(self, other):
        x0, y0, z0 = self.get_cartesian
        x1, y1, z1 = other.get_cartesian
        return Point3d(x=x0 - x1, y=y0 - y1, z=z0 - z1)

    def __mul__(self, other):
        x0, y0, z0 = self.get_cartesian
        x1, y1, z1 = other.get_cartesian
        return Point3d(x=x0 * x1, y=y0 * y1, z=z0 * z1)

    def dot(self, other) -> ufloat:
        x, y, z = (self * other).get_cartesian
        return x + y + z

    def angle(self, other):
        return self.dot(other) / (self.r * other.r)

    def distance(self, other):
        x0, y0, z0 = self.get_cartesian
        x1, y1, z1 = other.get_cartesian
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        return hpot(dx, dy, dz)

    @staticmethod
    def from_spherical(r, theta, phi):
        return Point3d(
            x=r * umath.cos(phi) * umath.sin(theta),
            y=r * umath.sin(phi) * umath.sin(theta),
            z=r * umath.cos(theta)
        )

    @staticmethod
    def nan():
        return Point3d(np.nan, np.nan, np.nan)


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
        return hpot(self.a, self.b, self.z)

    @property
    def constants(self):
        return self.a, self.b, self.c, self.p0.x, self.p0.y, self.p0.z

    def is_on_line(self, point: Point3d) -> bool:
        a, b, c, x0, y0, z0 = self.constants
        tx = (point.x - x0) / a
        ty = (point.y - x0) / b
        tz = (point.z - x0) / c
        return tx == ty and ty == tz

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


@dataclass
class Sphere:
    center: Point3d
    radius: ufloat

    def intercept_line3d(self, line: Line3d):
        a, b, c, x0, y0, z0 = line.constants
        dx, dy, dz = x0 - self.center.x, y0 - self.center.y, z0 - self.center.z

        A = line.v
        B = 2 * (a + b + c)
        C = (dx ** 2 + dy ** 2 + dz ** 2 - self.radius ** 2)

        det = B ** 2 - 4 * A * C
        if det < 0:
            return Point3d.nan(), Point3d.nan()

        t1 = (-B + umath.sqrt(det)) / (2 * A)
        t2 = (-B - umath.sqrt(det)) / (2 * A)

        return line.point_at(t1), line.point_at(t2)

    def is_on_sphere(self, point: Point3d) -> bool:
        dx, dy, dz = (self.center - point).get_cartesian
        return hpot(dx, dy, dz) == self.radius

    def arc_distance(self, p0: Point3d, p1: Point3d) -> ufloat:
        if not self.is_on_sphere(p0):
            raise ValueError(f'Point {p0} is not on the sphere {self}')
        if not self.is_on_sphere(p1):
            raise ValueError(f'Point {p1} is not on the sphere {self}')

        p2 = p0 - self.center
        p3 = p1 - self.center
        ang = p2.angle(p3)

        return ang * self.radius


@dataclass
class Degree:
    degree: int
    minutes: int
    secounds: int = 0
    fraction: int = 0

    def __repr__(self):
        return f'{self.degree}°{self.minutes}\'{self.secounds}.{self.fraction}\"'
    @property
    def error(self):
        if self.minutes == 0:
            return 1/60.0
        else:
            return 1/3600.0

    @property
    def angle(self):
        frac = float(f'{self.secounds}.{self.fraction}')
        return ufloat(
            nominal_value=self.degree+self.minutes/60.0+frac/3600.0,
            std_dev=self.error
        )

@dataclass
class Location:
    Latitude: Degree
    Longitude: Degree
    EarthRadius: ufloat = uncertainty_between(6_378_137.000, 6_356_752.314)


    @property
    def point(self):
        lat = self.Latitude.angle * np.pi / 180
        lng = self.Longitude.angle * np.pi / 180
        return Point3d.from_spherical(self.EarthRadius, lng, lat - np.pi / 2)


    @staticmethod
    def from_string(string: str):






class Earth:
    def __init__(self):
        self.sphere = Sphere(
            center=Point3d(0, 0, 0),
            radius=
        )

    def location(self, lat, lon):
