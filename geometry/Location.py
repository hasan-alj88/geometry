import numpy as np
import subprocess
import sys
from dataclasses import dataclass
from geometry.Degree import Degree
from geometry.UfloatOP import UfloatOP
from geometry.Point3d import Point3d
from geometry.Sphere import Sphere
from geometry.constant import EARTH_RADIUS

if 'uncertainties' not in sys.modules:
    subprocess.call(['pip', 'install', 'uncertainties'])

from uncertainties import ufloat
from uncertainties import unumpy as unp


@dataclass
class Location:
    Latitude: Degree
    Longitude: Degree
    EarthRadius: ufloat = EARTH_RADIUS

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
