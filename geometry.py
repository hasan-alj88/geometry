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




@dataclass
class Degree:
    degree: int
    minutes: int
    seconds: float = 0
    measurement_error: float = None

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
        if self.minutes == self.seconds == 0 and self.degree != 90:
            return 1 / 60.0 if self.measurement_error is None else self.measurement_error
        elif self.seconds == 0:
            return 1 / 3600.0 if self.measurement_error is None else self.measurement_error
        else:
            return 1 / 3600_000.0 if self.measurement_error is None else self.measurement_error

    @property
    def angle_degrees(self) -> ufloat:
        return ufloat(
            nominal_value=self.degree + self.minutes / 60.0 + self.seconds / 3600.0,
            std_dev=self.error
        )

    @property
    def angle_radian(self) -> ufloat:
        return self.angle_degrees * np.pi / 180.0

    @classmethod
    def from_string(cls, string: str):
        numbers_str = re.split(r'\D+', string)
        for _ in range(len(numbers_str) - 4):
            numbers_str.append("0")
        return cls(
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

    @classmethod
    def from_uflloat(cls, angle: ufloat, degrees: bool = True):
        full_circle = 360 if degrees else 2.0 * np.pi
        error = angle.std_dev
        ang = angle.nominal_value % full_circle
        ang = ang if degrees else ang * 180 / np.pi
        degree, minutes, seconds = Degree.sexagesimal(ang)
        return cls(
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

    @classmethod
    def from_string(cls, string: str):
        latitude_str, longitude_str = string.split('N')
        return cls(
            Latitude=Degree.from_string(latitude_str),
            Longitude=Degree.from_string(longitude_str)
        )

    @classmethod
    def from_float_tuple(cls, lat: ufloat, lng: ufloat, degree=True):
        return cls(
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

    @property
    def to_ufloat_tuple(self):
        lat, lon = self.to_float_tuple
        return ufloat(lat, self.Latitude.error), ufloat(lon, self.Longitude.error)

    def distance(self, other):
        """
         d = acos( sin φ1 ⋅ sin φ2 +
                   cos φ1 ⋅ cos φ2 ⋅ cos Δλ ) ⋅ R
         where 	φ is latitude, λ is longitude, R is earth’s radius
        :param other:
        :return:
        """
        lat1, lon1 = self.to_ufloat_tuple
        lat2, lon2 = other.to_ufloat_tuple
        dlon = lon2 - lon1
        return unp.arccos(unp.sin(lat1) * unp.sin(lat2) +
                          unp.cos(lat1) * unp.cos(lat2) * unp.cos(dlon)) * self.sphere.radius

    @property
    def url(self):
        lat, lon = self.to_float_tuple
        lat, lon = round(lat, 6), round(lon, 6)
        return f"https://www.google.com/maps/place/{lat},{lon}"

    def bearing_angle(self, other):
        lat1, lon1 = self.to_ufloat_tuple
        lat2, lon2 = other.to_ufloat_tuple
        dl = lat1 - lat2
        x = unp.cos(lon2) * unp.sin(dl)
        y = unp.cos(lon1) * unp.sin(lon2) - unp.sin(lon1) * unp.cos(lon2) * unp.cos(dl)
        return unp.arctan2(x / y)


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

    def mapping(self, location: Location) -> Point3d:
        angle = self.origin.bearing_angle(location)
        return Point3d(
            x=self.origin.distance(location) * unp.sin(angle),
            y=self.origin.distance(location) * unp.cos(angle),
            z=self.origin.sphere.radius
        )


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

    @property
    def tip(self) -> Point3d:
        return self.stick_line.after_by_length(point=self.base_point, length=self.length)


@dataclass
class SickShadowBySun:
    stick: Stick
    sun_location: Point3d

    @property
    def shadow_tip(self):
        line = Line3d(p0=self.sun_location, p1=self.stick.tip)
        shadow_p = self.stick.location.sphere.intercept_line3d(line=line)
        return self.stick.base_point.closest(list(shadow_p))


@dataclass
class SickShadowByMeasurement:
    stick: Stick
    shadow_length: ufloat
    shadow_angle: Degree

    @property
    def shadow_tip(self):
        shadow_trig = SphericalTriangle(
            r=self.stick.location.sphere.radius,
            c=self.shadow_length,
            gamma=ufloat(np.pi / 2, 0),
            alpha=(np.pi / 2) - self.shadow_angle.angle_radian
        )
        r, theta, phi = self.stick.base_point.get_spherical
        return Point3d.from_spherical(
            r=r,
            theta=theta + shadow_trig.b_radian,
            phi=phi + shadow_trig.a_radian
        )

    @property
    def shadow_line(self):
        return Line3d(p0=self.shadow_tip, p1=self.stick.tip)

    def sun_location(self, shadow2):
        line1 = self.shadow_line
        line2 = shadow2.shadow_line
        return line1.interception_line(line=line2)


sun = Sun()
print(sun.subsolar_now.url)
trig = SphericalTriangle(
    r=UfloatOP.from_minmax(6_378_137.000, 6_356_752.314),
    alpha=ufloat(np.pi / 2, 0),
    beta=ufloat(np.pi / 2, 0),
    gamma=ufloat(np.pi / 2, 0))
print(trig)
