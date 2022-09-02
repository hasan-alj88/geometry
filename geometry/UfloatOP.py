from uncertainties import ufloat
from uncertainties import unumpy as unp
from numpy import np


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
