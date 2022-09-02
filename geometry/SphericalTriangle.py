from geometry.UfloatOP import UfloatOP
from uncertainties import ufloat
from uncertainties import unumpy as unp
import numpy as np


def sin_rule_calc_side1(*, side1: ufloat, side2: ufloat, angle1: ufloat, angle2: ufloat):
    # sin side1 = sin angle1 . sin side2 / sin angle2
    if side1 is not None:
        return side1
    elif side1 is None and \
            angle1 is not None and \
            angle2 is not None and \
            side2 is not None:
        return unp.arcsin(unp.sin(angle1) * unp.sin(side2) / unp.sin(angle2))
    else:
        return None


def cos_rule_calc_side1(*, side1: ufloat, side2: ufloat, side3: ufloat, angle1: ufloat):
    # cos side1 = cos side2 . cos side3 + sin side2 . sin side3 . cos angle1
    if side1 is not None:
        return side1
    elif side1 is None and \
            side2 is not None and \
            side3 is not None and \
            angle1 is not None:
        return unp.arccos(unp.cos(side2) * unp.cos(side3) +
                          unp.sin(side2) * unp.sin(side3) * unp.cos(angle1))
    else:
        return None


def tan_rule(side1: ufloat, angle1: ufloat, angle2: ufloat, angle3: ufloat):
    # tan (side1 /2) = √( -cos(σ) cos(σ-angle1) / cos(σ-angle2) cos(σ-angle3) )
    # σ = (angle1+angle2+angle3)/2

    if side1 is not None:
        return side1
    elif side1 is None and \
            angle1 is not None and \
            angle2 is not None and \
            angle3 is not None:
        sigma = angle1 + angle2 + angle3
        nominator = - unp.cos(sigma) * unp.cos(sigma - angle1)
        denominator = unp.cos(sigma - angle2) * unp.cos(sigma - angle3)
        lhs = unp.sqrt(nominator / denominator)
        return 2 * unp.arctan(lhs)


def sin_rule1_right_angle_clac_side1(*, side1: ufloat, side3: ufloat, angle1: ufloat, angle3: ufloat):
    # sin a = sin alpha sin c if γ = pi/2
    if side1 is not None:
        return side1
    elif side1 is None and \
            side3 is not None and \
            angle1 is not None and \
            UfloatOP.is_in(np.pi / 2, angle3):
        return unp.acrsin(unp.sin(angle1) * unp.sin(side3))
    else:
        return None


def sin_rule2_right_angle_clac_side1(*, side1: ufloat, side2: ufloat, angle2: ufloat, angle3: ufloat):
    # sin b = sin β sin c = tan a cot alpha
    if side1 is not None:
        return side1
    elif side1 is None and \
            side2 is not None and \
            angle2 is not None and \
            UfloatOP.is_in(np.pi / 2, angle3):
        return unp.acrsin(unp.tan(side2) / unp.tan(angle2))
    else:
        return None


class SphericalTriangle:
    def __init__(self, *, r: ufloat, a: ufloat = None, b: ufloat = None, c: ufloat = None,
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

        for _ in range(6):
            #############
            # a_radian  #
            #############
            # sin a = sin alpha . sin b / sin β
            self.a_radian = sin_rule_calc_side1(
                side1=self.a_radian, side2=self.b_radian, angle1=self.alpha, angle2=self.beta)
            # sin a = sin alpha . sin c / sin γ
            self.a_radian = sin_rule_calc_side1(
                side1=self.a_radian, side2=self.c_radian, angle1=self.alpha, angle2=self.gamma)
            # cos a = cos b . cos c + sin a . sin c . cos alpha
            self.a_radian = cos_rule_calc_side1(
                side1=self.a_radian, side2=self.b_radian, side3=self.c_radian, angle1=self.alpha)
            # tan (a /2) = √( -cos(σ) cos(σ-alpha) / cos(σ-β) cos(σ-γ) ) ; σ = (alpha+β+γ)/2
            self.a_radian = tan_rule(side1=self.a_radian, angle1=self.alpha, angle2=self.beta, angle3=self.gamma)
            # sin a = sin alpha sin c if γ = pi/2
            self.a_radian = sin_rule1_right_angle_clac_side1(
                side1=self.a_radian, side3=self.c_radian, angle1=self.alpha, angle3=self.gamma)

            #############
            # b_radian  #
            #############
            # sin b = sin β . sin a / sin alpha
            self.b_radian = sin_rule_calc_side1(
                side1=self.b_radian, side2=self.a_radian, angle1=self.beta, angle2=self.alpha)
            # sin b = sin β . sin c / sin γ
            self.b_radian = sin_rule_calc_side1(
                side1=self.b_radian, side2=self.c_radian, angle1=self.beta, angle2=self.gamma)
            # cos b = cos a . cos c + sin a . sin c . cos β
            self.b_radian = cos_rule_calc_side1(
                side1=self.a_radian, side2=self.b_radian, side3=self.c_radian, angle1=self.beta)
            # tan (b /2) = √( -cos(σ) cos(σ-β) / cos(σ-alpha) cos(σ-γ) ) ; σ = (alpha+β+γ)/2
            self.b_radian = tan_rule(side1=self.b_radian, angle1=self.beta, angle2=self.alpha, angle3=self.gamma)
            # sin b = tan a cot α if γ=pi/2
            self.b_radian = sin_rule2_right_angle_clac_side1(
                side1=self.b_radian, side2=self.a_radian, angle2=self.alpha, angle3=self.gamma)

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
