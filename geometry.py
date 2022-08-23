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


class UfloatOP:
    @staticmethod
    def hypotenuse(x: ufloat, y: ufloat, z: ufloat):
        return unp.sqrt(x ** 2 + y ** 2 + z ** 2)

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
            return ufloat(np.nan, 0)

    @staticmethod
    def intersection_length(u_number1: ufloat, u_number2: ufloat) -> float:
        return UfloatOP.length(UfloatOP.intersection(u_number1, u_number2))

    @staticmethod
    def union_length(u_number1: ufloat, u_number2: ufloat) -> float:
        """
        |AUB| = |A| + |B| - |AnB|
        :param u_number1: uncertainty float A
        :param u_number2: uncertainty float B
        :return: length of |AUB|
        """
        return UfloatOP.length(u_number1) + \
               UfloatOP.length(u_number2) - \
               UfloatOP.intersection_length(u_number1, u_number2)

    @staticmethod
    def length(u_number: ufloat) -> float:
        if np.isnan(u_number.nominal_value):
            return 0.0
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

    @staticmethod
    def is_in(value: float, ufloat_range: ufloat) -> bool:
        minval, maxval = UfloatOP.get_min_max(ufloat_range)
        return minval <= value <= maxval


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
        return points[np.argmin(distances)]

    @classmethod
    def from_spherical(cls, r, theta, phi):
        return cls(
            x=r * unp.cos(phi) * unp.sin(theta),
            y=r * unp.sin(phi) * unp.sin(theta),
            z=r * unp.cos(theta)
        )

    @classmethod
    def nan(cls):
        return cls(np.nan, np.nan, np.nan)

    @classmethod
    def origin(cls):
        return cls(0, 0, 0)


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


class SphericalTriangle:
    def __init__(self, r: ufloat, a: ufloat = None, b: ufloat = None, c: ufloat = None,
                 alpha: ufloat = None, beta: ufloat = None, gamma: ufloat = None):
        self.r = r
        self.a_radian = a / r if a is not None else None
        self.b_radian = b / r if b is not None else None
        self.c_radian = c / r if c is not None else None
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        if r is None:
            raise ValueError(f'sphere radius not given')

        for _ in range(7):
            #############
            # a_radian  #
            #############
            # sin a = sin alpha . sin b / sin β
            if self.a_radian is None and \
                    self.alpha is not None and \
                    self.beta is not None and \
                    self.b_radian is not None:
                self.a_radian = unp.arcsin(unp.sin(self.alpha) * unp.sin(self.b_radian) / unp.sin(self.beta))

            # sin a = sin alpha . sin c / sin γ
            elif self.a_radian is None and \
                    self.alpha is not None and \
                    self.gamma is not None and \
                    self.c_radian is not None:
                self.a_radian = unp.arcsin(unp.sin(self.alpha) * unp.sin(self.c_radian) / unp.sin(self.gamma))

            # cos a = cos b . cos c + sin b . sin c . cos alpha
            elif self.a_radian is None and \
                    self.b_radian is not None and \
                    self.c_radian is not None and \
                    self.alpha is not None:
                self.a_radian = unp.arccos(unp.cos(self.b_radian) * unp.cos(self.c_radian) +
                                           unp.sin(self.b_radian) * unp.sin(self.c_radian) * unp.cos(self.alpha))

            # tan (a /2) = √( -cos(σ) cos(σ-α) / cos(σ-β) cos(σ-γ) )
            # σ = (alpha+β+γ)/2
            elif self.a_radian is None and \
                    self.alpha is not None and \
                    self.beta is not None and \
                    self.gamma is not None:
                sigma = (self.alpha + self.beta + self.gamma) / 2
                tan_a_sqr = -1 * unp.cos(sigma) * unp.cos(sigma - self.alpha) / \
                            (np.cos(sigma - self.beta) * unp.cos(sigma - self.gamma))
                self.a_radian = unp.arctan(np.sqrt(tan_a_sqr))

            # sin a = sin alpha sin c if γ = pi/2
            elif self.a_radian is None and \
                    self.alpha is not None and \
                    self.c_radian is not None and \
                    UfloatOP.is_in(np.pi / 2, self.gamma):
                self.a_radian = unp.arcsin(unp.sin(self.alpha) * unp.sin(self.c_radian))

            # sin a = tan b cot β if γ = pi/2
            elif self.a_radian is None and \
                    self.b_radian is not None and \
                    self.beta is not None and \
                    UfloatOP.is_in(np.pi / 2, self.gamma):
                self.a_radian = np.arcsin(np.tan(self.b_radian) / np.tan(self.beta))

            #############
            # b_radian  #
            #############
            # sin b = sin β . sin a / sin alpha
            if self.b_radian is None and \
                    self.beta is not None and \
                    self.alpha is not None and \
                    self.a_radian is not None:
                self.b_radian = unp.arcsin(unp.sin(self.beta) * unp.sin(self.a_radian) / unp.sin(self.alpha))

            # sin b = sin β . sin c / sin γ
            elif self.b_radian is None and \
                    self.beta is not None and \
                    self.gamma is not None and \
                    self.c_radian is not None:
                self.b_radian = unp.arcsin(unp.sin(self.beta) * unp.sin(self.c_radian) / unp.sin(self.gamma))

            # cos b = cos a . cos c + sin a . sin c . cos β
            elif self.b_radian is None and \
                    self.a_radian is not None and \
                    self.c_radian is not None and \
                    self.beta is not None:
                self.b_radian = unp.arccos(unp.cos(self.a_radian) * unp.cos(self.c_radian) +
                                           unp.sin(self.a_radian) * unp.sin(self.c_radian) * unp.cos(self.beta))

            # tan (b /2) = √( -cos(σ) cos(σ-β) / cos(σ-alpha) cos(σ-γ) )
            # σ = (alpha+β+γ)/2
            elif self.b_radian is None and \
                    self.alpha is not None and \
                    self.beta is not None and \
                    self.gamma is not None:
                sigma = (self.alpha + self.beta + self.gamma) / 2
                tan_b_sqr = -1 * unp.cos(sigma) * unp.cos(sigma - self.beta) / \
                            (unp.cos(sigma - self.alpha) * unp.cos(sigma - self.gamma))
                self.a_radian = unp.arctan(unp.sqrt(tan_b_sqr))

            # sin b = sin β sin c if γ=pi/2
            elif self.b_radian is None and \
                    self.beta is not None and \
                    self.c_radian is not None and \
                    UfloatOP.is_in(np.pi / 2, self.gamma):
                self.b_radian = unp.arcsin(unp.sin(self.beta) * unp.sin(self.c_radian))

            # sin b = tan a cot α if γ=pi/2
            elif self.b_radian is None and \
                    self.a_radian is not None and \
                    self.alpha is not None and \
                    UfloatOP.is_in(np.pi / 2, self.gamma):
                self.b_radian = unp.arcsin(unp.tan(self.a_radian) / unp.tan(self.alpha))

            #############
            # c_radian  #
            #############
            # sin c = sin γ . sin a / sin alpha
            if self.c_radian is None and \
                    self.gamma is not None and \
                    self.alpha is not None and \
                    self.a_radian is not None:
                self.c_radian = unp.arcsin(unp.sin(self.gamma) * unp.sin(self.a_radian) / unp.sin(self.alpha))

            # sin c = sin γ . sin b / sin β
            elif self.c_radian is None and \
                    self.gamma is not None and \
                    self.beta is not None and \
                    self.b_radian is not None:
                self.c_radian = unp.arcsin(unp.sin(self.gamma) * unp.sin(self.b_radian) / unp.sin(self.beta))

            # cos c = cos a . cos b + sin a . sin b . cos γ
            elif self.c_radian is None and \
                    self.a_radian is not None and \
                    self.b_radian is not None and \
                    self.gamma is not None:
                self.c_radian = unp.arccos(unp.cos(self.a_radian) * unp.cos(self.b_radian) +
                                           unp.sin(self.a_radian) * unp.sin(self.b_radian) * unp.cos(self.gamma))

            # tan (c /2) = √( -cos(σ) cos(σ-γ) / cos(σ-alpha) cos(σ-β) )
            # σ = (alpha+β+γ)/2
            elif self.c_radian is None and \
                    self.alpha is not None and \
                    self.beta is not None and \
                    self.gamma is not None:
                sigma = (self.alpha + self.beta + self.gamma) / 2
                tan_c_sqr = -1 * unp.cos(sigma) * unp.cos(sigma - self.gamma) / \
                            (unp.cos(sigma - self.alpha) * unp.cos(sigma - self.beta))
                self.c_radian = unp.arctan(unp.sqrt(tan_c_sqr))

            ##########
            # alpha  #
            ##########
            # sin alpha = sin a . sin β / sin b
            if self.alpha is None and \
                    self.a_radian is not None and \
                    self.beta is not None and \
                    self.b_radian is not None:
                self.alpha = unp.arcsin(unp.sin(self.a_radian) * unp.sin(self.beta) / unp.sin(self.b_radian))

            # sin alpha = sin a . sin γ / sin c
            elif self.alpha is None and \
                    self.a_radian is not None and \
                    self.gamma is not None and \
                    self.c_radian is not None:
                self.alpha = unp.arcsin(unp.sin(self.a_radian) * unp.sin(self.gamma) / unp.sin(self.c_radian))

            # cos alpha = -cos β cos γ + sin β sin γ cos a
            elif self.alpha is None and \
                    self.beta is not None and \
                    self.gamma is not None and \
                    self.a_radian is not None:
                self.alpha = unp.arccos(-unp.cos(self.beta) * unp.cos(self.gamma) +
                                        unp.sin(self.beta) * unp.sin(self.gamma) * unp.cos(self.a_radian))

            # tan (alpha/2) = √( sin(s–b) sin(s–c) / sin s sin(s–a) )
            # s = (a+b+c)/2
            elif self.alpha is None and \
                    self.a_radian is not None and \
                    self.b_radian is not None and \
                    self.c_radian is not None:
                s = (self.a_radian + self.b_radian + self.c_radian) / 2
                tan_alpha_sqr = unp.sin(s - self.b_radian) * unp.sin(s - self.c_radian) / \
                                (np.sin(s) * unp.sin(s - self.a_radian))
                self.alpha = unp.arctan(np.sqrt(tan_alpha_sqr))

            ##########
            # Beta  #
            ##########
            # sin β = sin b . sin alpha / sin a
            if self.beta is None and \
                    self.b_radian is not None and \
                    self.alpha is not None and \
                    self.a_radian is not None:
                self.beta = unp.arcsin(unp.sin(self.b_radian) * unp.sin(self.alpha) / unp.sin(self.a_radian))

            # sin β = sin b . sin γ / sin c
            elif self.beta is None and \
                    self.b_radian is not None and \
                    self.gamma is not None and \
                    self.c_radian is not None:
                self.beta = unp.arcsin(unp.sin(self.b_radian) * unp.sin(self.gamma) / unp.sin(self.c_radian))

            # cos β = -cos alpha cos γ + sin alpha sin γ cos b
            elif self.beta is None and \
                    self.alpha is not None and \
                    self.gamma is not None and \
                    self.b_radian is not None:
                self.beta = unp.arccos(-unp.cos(self.alpha) * unp.cos(self.gamma) +
                                       unp.sin(self.alpha) * unp.sin(self.gamma) * unp.cos(self.b_radian))

            # tan (β/2) = √( sin(s–a) sin(s–c) / sin s sin(s–b) )
            # s = (a+b+c)/2
            elif self.beta is None and \
                    self.a_radian is not None and \
                    self.b_radian is not None and \
                    self.c_radian is not None:
                s = (self.a_radian + self.b_radian + self.c_radian) / 2
                tan_beta_sqr = unp.sin(s - self.a_radian) * unp.sin(s - self.b_radian) / \
                               (np.sin(s) * unp.sin(s - self.b_radian))
                self.beta = unp.arctan(np.sqrt(tan_beta_sqr))

            ##########
            # gamma  #
            ##########
            # sin γ = sin c . sin alpha / sin c
            if self.gamma is None and \
                    self.c_radian is not None and \
                    self.alpha is not None and \
                    self.a_radian is not None:
                self.gamma = unp.arcsin(unp.sin(self.c_radian) * unp.sin(self.alpha) / unp.sin(self.a_radian))

            # sin γ = sin c . sin β / sin b
            elif self.gamma is None and \
                    self.c_radian is not None and \
                    self.beta is not None and \
                    self.b_radian is not None:
                self.gamma = unp.arcsin(unp.sin(self.c_radian) * unp.sin(self.beta) / unp.sin(self.b_radian))

            # cos γ = -cos alpha cos β + sin alpha sin β cos c
            elif self.gamma is None and \
                    self.alpha is not None and \
                    self.beta is not None and \
                    self.c_radian is not None:
                self.gamma = unp.arccos(-unp.cos(self.alpha) * unp.cos(self.beta) +
                                        unp.sin(self.alpha) * unp.sin(self.beta) * unp.cos(self.c_radian))

            # tan (γ/2) = √( sin(s–b) sin(s–a) / sin s sin(s–c) )
            # s = (a+b+c)/2
            elif self.gamma is None and \
                    self.a_radian is not None and \
                    self.b_radian is not None and \
                    self.c_radian is not None:
                s = (self.a_radian + self.b_radian + self.c_radian) / 2
                tan_gamma_sqr = unp.sin(s - self.b_radian) * unp.sin(s - self.a_radian) / \
                                (np.sin(s) * unp.sin(s - self.c_radian))
                self.gamma = unp.arctan(np.sqrt(tan_gamma_sqr))

    def __repr__(self):
        out = f'r = {self.r}\n'
        out += f'a = {self.a}\n'
        out += f'b = {self.b}\n'
        out += f'c = {self.c}\n'
        out += f'a_rad = {self.a_radian}\n'
        out += f'b_rad = {self.b_radian}\n'
        out += f'c_rad = {self.c_radian}\n'
        out += f'alpha = {self.alpha}\n'
        out += f'β = {self.beta}\n'
        out += f'γ = {self.gamma}\n'
        return out

    @property
    def no_nan(self) -> bool:
        return self.a is not None and \
               self.b is not None and \
               self.c is not None and \
               self.alpha is not None and \
               self.beta is not None and \
               self.gamma is not None

    @property
    def a(self):
        return self.a_radian * self.r if self.a_radian is not None else None

    @property
    def b(self):
        return self.b_radian * self.r if self.b_radian is not None else None

    @property
    def c(self):
        return self.c_radian * self.r if self.c_radian is not None else None


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
        trig = SphericalTriangle(
            r=self.sphere.radius,
            c=self.origin.distance(location),
            gamma=ufloat(np.pi / 2, 0),
        )
        return Point3d(
            x=trig.b,
            y=trig.a,
            z=self.sphere.radius
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
            theta=theta+shadow_trig.b_radian,
            phi=phi+shadow_trig.a_radian
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
    alpha=ufloat(0.1, 1e-3),
    c=ufloat(1, 1e-3),
    gamma=ufloat(np.pi / 2, 0))
print(trig)
