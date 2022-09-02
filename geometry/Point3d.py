from geometry.UfloatOP import UfloatOP
from uncertainties import unumpy as unp
from uncertainties import ufloat
from numpy import nan, argmin
from dataclasses import dataclass


@dataclass
class Point3d:
    x: ufloat
    y: ufloat
    z: ufloat

    @property
    def r(self):
        return UfloatOP.hypotenuse(self.x, self.y, self.z)

    @property
    def theta(self) -> ufloat:
        return unp.cos(self.z / self.r)

    @property
    def phi(self):
        return unp.arctan2(self.x / self.y)

    @property
    def get_cartesian(self):
        return self.x, self.y, self.z

    @property
    def get_spherical(self):
        return self.r, self.theta, self.phi

    @property
    def normalized(self):
        r, theta, phi = self.get_spherical
        return Point3d.from_spherical(r=1, theta=theta, phi=phi)

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
        return unp.arccos(self.dot(other) / (self.r * other.r))

    def distance(self, other):
        x0, y0, z0 = self.get_cartesian
        x1, y1, z1 = other.get_cartesian
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        return UfloatOP.hypotenuse(dx, dy, dz)

    def closest(self, points):
        distances = [self.distance(point) for point in points]
        return points[argmin(distances)]

    @classmethod
    def from_spherical(cls, r, theta, phi):
        return cls(
            x=r * unp.cos(phi) * unp.sin(theta),
            y=r * unp.sin(phi) * unp.sin(theta),
            z=r * unp.cos(theta)
        )

    @classmethod
    def nan(cls):
        return cls(nan, nan, nan)

    @classmethod
    def origin(cls):
        return cls(0, 0, 0)
