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
from uncertainties import umath


class UfloatOP:
    @staticmethod
    def hypotenuse(x: ufloat, y: ufloat, z: ufloat):
        return umath.sqrt(x ** 2 + y ** 2 + z ** 2)

    @staticmethod
    def from_minmax(maxval: float, minval: float):
        return ufloat((maxval + minval) / 2, abs(maxval - minval) / 2)

    @staticmethod
    def get_min_max(u_number: ufloat):
        return u_number.nominal_value - u_number.std_dev, u_number.nominal_value + u_number.std_dev

    @staticmethod
    def is_subset(u_number1: ufloat, u_number2: ufloat) -> bool:
        x_min, x_max = UfloatOP.get_min_max(u_number1)
        y_min, y_max = UfloatOP.get_min_max(u_number2)
        return y_min <= x_min <= x_max <= y_max

    @staticmethod
    def intersection(u_number1: ufloat, u_number2: ufloat) -> ufloat:
        x_min, x_max = UfloatOP.get_min_max(u_number1)
        y_min, y_max = UfloatOP.get_min_max(u_number2)
        if UfloatOP.is_subset(u_number1, u_number2):
            return u_number1
        elif UfloatOP.is_subset(u_number2, u_number1):
            return u_number2
        elif y_min < x_min < y_max < x_max:
            return UfloatOP.from_minmax(x_min, y_max)
        elif x_min < y_min < x_max < y_max:
            return UfloatOP.from_minmax(y_min, x_max)
        else:
            return ufloat(0, 0)

    @staticmethod
    def intersection_length(u_number1: ufloat, u_number2: ufloat) -> float:
        return UfloatOP.length(UfloatOP.intersection(u_number1, u_number2))

    @staticmethod
    def union_length(u_number1: ufloat, u_number2: ufloat) -> float:
        return UfloatOP.length(u_number1) + \
               UfloatOP.length(u_number2) - \
               UfloatOP.intersection_length(u_number1, u_number2)

    @staticmethod
    def length(u_number: ufloat) -> float:
        x_min, x_max = UfloatOP.get_min_max(u_number)
        return abs(x_max - x_min)

    @staticmethod
    def accuracy(u_number1: ufloat, u_number2: ufloat) -> float:
        aub = UfloatOP.union_length(u_number1, u_number2)
        anb = UfloatOP.intersection_length(u_number1, u_number2)
        if aub == 0:  # Exact measurement
            return 1.0 if u_number1 == u_number2 else 0
        elif UfloatOP.is_subset(u_number1, u_number2):  # if it's a subset then prediction is accurate
            return 1.0
        else:  # if not a subset then accuracy depend on overlap (if any)
            return aub / anb


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
        return umath.arccos(self.dot(other) / (self.r * other.r))

    def distance(self, other):
        x0, y0, z0 = self.get_cartesian
        x1, y1, z1 = other.get_cartesian
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        return UfloatOP.hypotenuse(dx, dy, dz)

    def closest(self, points):
        distances = [self.distance(point) for point in points]
        return points[np.argmin(distances)]

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

    @staticmethod
    def origin():
        return Point3d(0, 0, 0)


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

    @staticmethod
    def from_point_and_vector(point: Point3d, vector: Point3d):
        return Line3d(
            p0=point,
            p1=point + vector.normalized
        )


@dataclass
class Sphere:
    radius: ufloat
    center: Point3d = Point3d.origin()

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
        return UfloatOP.hypotenuse(dx, dy, dz) == self.radius

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
    seconds: float = 0
    measurement_error: float = 0

    def __repr__(self):
        e_degree, e_minutes, e_seconds = Degree.sexagesimal(self.error)
        error_str = '±'
        if e_degree > 0:
            error_str += f'{e_degree}°'
        if e_minutes > 0:
            error_str += f'{e_minutes}\''
        if e_seconds > 0:
            error_str += f'{e_seconds}\"'
        if error_str == '±':
            error_str = ""
        return f' {self.degree}°{self.minutes}\'{self.seconds:.6f}\"{error_str} '

    def __add__(self, other):
        ang1 = self.angle_degrees(self.angle_degrees, degrees=True)
        ang2 = self.angle_degrees(other.angle_degrees, degrees=True)
        return Degree.from_uflloat((ang1 + ang2) % 360, degrees=True)

    def __sub__(self, other):
        ang1 = self.angle_degrees(self.angle_degrees, degrees=True) % 360
        ang2 = self.angle_degrees(other.angle_degrees, degrees=True) % 360
        return Degree.from_uflloat((ang1 - ang2) % 360, degrees=True)

    @property
    def error(self):
        if self.minutes == 0 and self.seconds == 0:
            return max(1 / 60.0, self.measurement_error)
        elif self.seconds == 0:
            return max(1 / 3600.0, self.measurement_error)
        else:
            return max(1 / 3600_000.0, self.measurement_error)

    @property
    def angle_degrees(self) -> ufloat:
        return ufloat(
            nominal_value=self.degree + self.minutes / 60.0 + self.seconds / 3600.0,
            std_dev=self.error
        )

    @property
    def angle_radian(self) -> ufloat:
        return self.angle_degrees * np.pi / 180.0

    @staticmethod
    def from_string(string: str):
        numbers_str = re.split(r'\D+', string)
        for _ in range(len(numbers_str) - 4):
            numbers_str.append("0")
        return Degree(
            degree=int(numbers_str[0]),
            minutes=int(numbers_str[0]),
            seconds=float(numbers_str[0])
        )

    @staticmethod
    def sexagesimal(number: float):
        degree = int(number)
        minutes = int((number - degree) * 60)
        seconds = number - (degree + minutes / 60.0)
        return degree, minutes, seconds

    @staticmethod
    def from_uflloat(angle: ufloat, degrees: bool = True):
        full_circle = 360 if degrees else 2.0 * np.pi
        error = angle.std_dev
        ang = angle.nominal_value % full_circle
        ang = ang if degrees else ang * 180 / np.pi
        degree, minutes, seconds = Degree.sexagesimal(ang)
        return Degree(
            degree=degree,
            minutes=minutes,
            seconds=seconds,
            measurement_error=error
        )


@dataclass
class Location:
    Latitude: Degree
    Longitude: Degree
    EarthRadius: ufloat = UfloatOP.from_minmax(6_378_137.000, 6_356_752.314)

    def __repr__(self):
        return f'({self.Latitude} N {self.Longitude} E)'

    @property
    def point(self):
        lat = self.Latitude.angle_radian
        lng = self.Longitude.angle_radian
        return Point3d.from_spherical(self.EarthRadius, lng, lat - np.pi / 2)

    @property
    def sphere(self):
        return Sphere(radius=self.EarthRadius)

    @staticmethod
    def from_string(string: str):
        latitude_str, longitude_str = string.split('N')
        return Location(
            Latitude=Degree.from_string(latitude_str),
            Longitude=Degree.from_string(longitude_str)
        )

    @staticmethod
    def from_float_tuple(lat: ufloat, lng: ufloat, degree=True):
        return Location(
            Latitude=Degree.from_uflloat(lat, degree),
            Longitude=Degree.from_uflloat(lng, degree)
        )

    @property
    def to_float_tuple(self):
        lat, lon = self.Latitude.angle_degrees.nominal_value, self.Longitude.angle_degrees.nominal_value
        lat = lat if -90 < lat < 90 else lat % 360
        lat = lat if -90 < lat < 90 else lat - 360
        lon = lon if -180 < lon < 180 else lon % 360
        lon = lon if -180 < lon < 180 else lon - 360
        return lat, lon

    def distance(self, other):
        p0 = self.point
        p1 = other.point
        return self.sphere.arc_distance(p0, p1)

    @property
    def url(self):
        lat, lon = self.to_float_tuple
        lat, lon = round(lat, 6), round(lon, 6)
        return f"https://www.google.com/maps/place/{lat},{lon}"


@dataclass
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


@dataclass
class FlattenSphere:
    origin: Location
    sphere: Sphere

    @property
    def plane(self):
        return Plane3d.from_point_and_normal(self.origin.point, self.origin.point - self.sphere.center)

    def projection(self, point: Point3d) -> Point3d:
        line = Line3d.from_point_and_vector(point=point, vector=self.plane.normal)
        return self.plane.line3d_interception(line=line)


sun = Sun()
print(sun.subsolar_now.to_float_tuple)
print(sun.subsolar_now.url)
