import numpy as np
import subprocess
import sys
from dataclasses import dataclass
import re

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
