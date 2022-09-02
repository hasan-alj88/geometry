from geometry.UfloatOP import UfloatOP
from geometry.Line3d import Line3d
from geometry.Point3d import Point3d
from uncertainties import ufloat
from uncertainties import unumpy as unp
import numpy as np
from dataclasses import dataclass


class FlatTriangle:
    def __init__(self, *, a: ufloat, b: ufloat, c: ufloat, alpha: ufloat, beta: ufloat, gamma: ufloat):
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        for _ in range(6):
            self.a = FlatTriangle.sin_rule_clac_side1(side1=self.a, side2=self.b, angle1=self.alpha, angle2=self.beta)
            self.a = FlatTriangle.sin_rule_clac_side1(side1=self.a, side2=self.c, angle1=self.alpha, angle2=self.gamma)
            self.a = FlatTriangle.cos_rule_clac_side3(side1=self.b, side2=self.c, side3=self.a, angle3=self.alpha)

            self.b = FlatTriangle.sin_rule_clac_side1(side1=self.b, side2=self.a, angle1=self.beta, angle2=self.alpha)
            self.b = FlatTriangle.sin_rule_clac_side1(side1=self.b, side2=self.c, angle1=self.beta, angle2=self.gamma)
            self.b = FlatTriangle.cos_rule_clac_side3(side1=self.a, side2=self.c, side3=self.b, angle3=self.beta)

            self.c = FlatTriangle.sin_rule_clac_side1(side1=self.c, side2=self.a, angle1=self.gamma, angle2=self.alpha)
            self.c = FlatTriangle.sin_rule_clac_side1(side1=self.c, side2=self.b, angle1=self.gamma, angle2=self.beta)
            self.c = FlatTriangle.cos_rule_clac_side3(side1=self.a, side2=self.b, side3=self.c, angle3=self.gamma)

            self.alpha = FlatTriangle.sin_rule_clac_angle1(side1=self.a, side2=self.b, angle1=self.alpha,
                                                           angle2=self.beta)
            self.alpha = FlatTriangle.sin_rule_clac_angle1(side1=self.a, side2=self.c, angle1=self.alpha,
                                                           angle2=self.gamma)
            self.alpha = FlatTriangle.cos_rule_clac_angle3(side1=self.b, side2=self.c, side3=self.a, angle3=self.alpha)

            self.beta = FlatTriangle.sin_rule_clac_angle1(side1=self.b, side2=self.a, angle1=self.beta,
                                                          angle2=self.alpha)
            self.beta = FlatTriangle.sin_rule_clac_angle1(side1=self.b, side2=self.c, angle1=self.beta,
                                                          angle2=self.gamma)
            self.beta = FlatTriangle.cos_rule_clac_angle3(side1=self.a, side2=self.c, side3=self.b, angle3=self.beta)

            self.gamma = FlatTriangle.sin_rule_clac_angle1(side1=self.c, side2=self.a, angle1=self.gamma,
                                                           angle2=self.alpha)
            self.gamma = FlatTriangle.sin_rule_clac_angle1(side1=self.c, side2=self.b, angle1=self.gamma,
                                                           angle2=self.beta)
            self.gamma = FlatTriangle.cos_rule_clac_angle3(side1=self.a, side2=self.c, side3=self.b, angle3=self.gamma)

    @staticmethod
    def sin_rule_clac_side1(*, side1: ufloat, side2: ufloat, angle1: ufloat, angle2: ufloat):
        if side1 is not None:
            return side1
        elif side1 is None and \
                angle1 is not None and \
                side2 is not None and \
                angle2 is not None:
            return side2 * unp.sin(angle1) / unp.sin(angle2)
        else:
            return None

    @staticmethod
    def sin_rule_clac_angle1(*, side1: ufloat, side2: ufloat, angle1: ufloat, angle2: ufloat):
        if angle1 is not None:
            return angle1
        elif angle1 is None and \
                side1 is not None and \
                side2 is not None and \
                angle2 is not None:
            return unp.arcsin(unp.sin(angle2) * side1 / side2)
        else:
            return None

    @staticmethod
    def cos_rule_clac_side3(*, side1: ufloat, side2: ufloat, side3: ufloat, angle3: ufloat):
        if side3 is not None:
            return side3
        elif side3 is None and \
                side1 is not None and \
                side2 is not None and \
                angle3 is not None:
            return side1 ** 2 + side2 ** 2 - 2 * side1 * side2 * unp.cos(angle3)
        else:
            return None

    @staticmethod
    def cos_rule_clac_angle3(*, side1: ufloat, side2: ufloat, side3: ufloat, angle3: ufloat):
        if angle3 is not None:
            return angle3
        elif angle3 is None and \
                side1 is not None and \
                side2 is not None and \
                side3 is not None:
            return unp.arccos((side1 ** 2 + side2 ** 2 - side3 **2) / (2 * side1 * side2))
