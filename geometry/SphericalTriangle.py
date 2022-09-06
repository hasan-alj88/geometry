from geometry.UfloatOP import UfloatOP
from uncertainties import ufloat
from uncertainties import unumpy as unp
import numpy as np


def sin_rule_calc_top1(*, top1: ufloat, top2: ufloat, down1: ufloat, down2: ufloat):
    # sin top1 = sin down1 . sin top2 / sin down1
    if top1 is not None:
        return top1
    elif top1 is None and \
            down1 is not None and \
            down2 is not None and \
            top2 is not None:
        return unp.arcsin(unp.sin(down1) * unp.sin(top2) / unp.sin(down2))
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


def cos_rule_calc_angle1(*, angle1: ufloat, angle2: ufloat, angle3: ufloat, side1: ufloat):
    # cos angle1 = -cos angle2 cos angle3 + sin angle2 sin angle3 cos side1
    if angle1 is not None:
        return angle1
    elif angle1 is None and \
            angle2 is not None and \
            angle3 is not None and \
            side1 is not None:
        return unp.arccos(- unp.cos(angle2) * unp.cos(angle3) +
                          unp.sin(angle2) * unp.sin(angle3) * unp.cos(side1))
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


def right_sin_sin_sin_rule(*, lhs, sin_1_arg, sin_2_arg, right_angle):
    # sin lhs = sin sin_1_arg . sin sin_2_arg if right_angle = pi/2
    if lhs is not None:
        return lhs
    elif lhs is None and sin_1_arg is not None and sin_2_arg is not None and UfloatOP.is_in(np.pi / 2, right_angle):
        return unp.arcsin(unp.sin(sin_1_arg) * unp.sin(sin_2_arg))
    else:
        return None


def right_sin_tan_cot_rule(*, lhs, tan_arg, cot_arg, right_angle):
    # sin lhs = tan tan_arg . cot cot_arg if right_angle = pi/2
    if lhs is not None:
        return lhs
    elif lhs is None and tan_arg is not None and cot_arg is not None and UfloatOP.is_in(np.pi / 2, right_angle):
        return unp.arcsin(unp.tan(tan_arg) / unp.tan(cot_arg))
    else:
        return None


def right_cos_cos_sin(*, lhs, cos_arg, sin_arg, right_angle):
    # cos lhs = cos cos_arg . sin sin_arg if right_angle = pi/2
    if lhs is not None:
        return lhs
    elif lhs is None and cos_arg is not None and sin_arg is not None and UfloatOP.is_in(np.pi / 2, right_angle):
        return unp.arccos(unp.cos(cos_arg) * unp.sin(sin_arg))
    else:
        return None


def right_cos_tan_cot_rule(*, lhs, tan_arg, cot_arg, right_angle):
    # cos lhs = tan tan_arg . cot cot_arg if right_angle = pi/2
    if lhs is not None:
        return lhs
    elif lhs is None and tan_arg is not None and cot_arg is not None and UfloatOP.is_in(np.pi / 2, right_angle):
        return unp.arccos(unp.tan(tan_arg) / unp.tan(cot_arg))
    else:
        return None


def right_cos_cos_cos_rule(*, lhs, cos_1_arg, cos_2_arg, right_angle):
    # cos lhs = cos cos_1_arg . cos cos_2_arg if right_angle = pi/2
    if lhs is not None:
        return lhs
    elif lhs is None and cos_1_arg is not None and cos_2_arg is not None and UfloatOP.is_in(np.pi / 2, right_angle):
        return unp.arcos(unp.cos(cos_1_arg) * unp.cos(cos_2_arg))
    else:
        return None


def right_cos_cot_cot_rule(*, lhs, cot_1_arg, cot_2_arg, right_angle):
    # cos lhs = cot cot_1_arg . cot cot_2_arg if right_angle = pi/2
    if lhs is not None:
        return lhs
    elif lhs is None and cot_1_arg is not None and cot_2_arg is not None and UfloatOP.is_in(np.pi / 2, right_angle):
        return unp.arcos(1 / (unp.tan(cot_1_arg) * unp.tan(cot_2_arg)))
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
            self.a_radian = sin_rule_calc_top1(
                top1=self.a_radian, top2=self.b_radian, down1=self.alpha, down2=self.beta)
            # sin a = sin alpha . sin c / sin γ
            self.a_radian = sin_rule_calc_top1(
                top1=self.a_radian, top2=self.c_radian, down1=self.alpha, down2=self.gamma)
            # cos a = cos b . cos c + sin a . sin c . cos alpha
            self.a_radian = cos_rule_calc_side1(
                side1=self.a_radian, side2=self.b_radian, side3=self.c_radian, angle1=self.alpha)
            # tan (a /2) = √( -cos(σ) cos(σ-alpha) / cos(σ-β) cos(σ-γ) ) ; σ = (alpha+β+γ)/2
            self.a_radian = tan_rule(side1=self.a_radian, angle1=self.alpha, angle2=self.beta, angle3=self.gamma)
            # sin a = sin alpha sin c if γ = pi/2
            self.a_radian = right_sin_sin_sin_rule(
                lhs=self.a_radian, sin_1_arg=self.alpha, sin_2_arg=self.c_radian, right_angle=self.gamma)
            # sin a = tan b cot β if γ = pi/2
            self.a_radian = right_sin_tan_cot_rule(
                lhs=self.a_radian, tan_arg=self.b_radian, cot_arg=self.beta, right_angle=self.gamma)

            #############
            # b_radian  #
            #############
            # sin b = sin β . sin a / sin alpha
            self.b_radian = sin_rule_calc_top1(
                top1=self.b_radian, top2=self.a_radian, down1=self.beta, down2=self.alpha)
            # sin b = sin β . sin c / sin γ
            self.b_radian = sin_rule_calc_top1(
                top1=self.b_radian, top2=self.c_radian, down1=self.beta, down2=self.gamma)
            # cos b = cos a . cos c + sin a . sin c . cos β
            self.b_radian = cos_rule_calc_side1(
                side1=self.a_radian, side2=self.b_radian, side3=self.c_radian, angle1=self.beta)
            # tan (b /2) = √( -cos(σ) cos(σ-β) / cos(σ-alpha) cos(σ-γ) ) ; σ = (alpha+β+γ)/2
            self.b_radian = tan_rule(side1=self.b_radian, angle1=self.beta, angle2=self.alpha, angle3=self.gamma)
            # sin b = sin β sin c if γ = pi/2
            self.b_radian = right_sin_sin_sin_rule(
                lhs=self.b_radian, sin_1_arg=self.beta, sin_2_arg=self.c_radian, right_angle=self.gamma)
            # sin b = tan a cot α  if γ = pi/2
            self.a_radian = right_sin_tan_cot_rule(
                lhs=self.b_radian, tan_arg=self.a_radian, cot_arg=self.alpha, right_angle=self.gamma)

            #############
            # c_radian  #
            #############
            # sin c = sin γ . sin a / sin alpha
            self.c_radian = sin_rule_calc_top1(
                top1=self.c_radian, top2=self.a_radian, down1=self.gamma, down2=self.alpha)
            # sin c = sin γ . sin b / sin β
            self.c_radian = sin_rule_calc_top1(
                top1=self.c_radian, top2=self.b_radian, down1=self.gamma, down2=self.beta)
            # cos c = cos a . cos b + sin a . sin b . cos γ
            self.c_radian = cos_rule_calc_side1(
                side1=self.a_radian, side2=self.b_radian, side3=self.c_radian, angle1=self.gamma)
            # tan (c /2) = √( -cos(σ) cos(σ-γ) / cos(σ-alpha) cos(σ-β) ) ; σ = (alpha+β+γ)/2
            self.c_radian = tan_rule(
                side1=self.b_radian, angle1=self.gamma, angle2=self.alpha, angle3=self.beta)
            # cos c = cot alpha cot β if γ = pi/2
            self.c_radian = right_cos_cot_cot_rule(
                lhs=self.c_radian, cot_1_arg=self.alpha, cot_2_arg=self.beta, right_angle=self.gamma)
            # cos c = cos a cos b if γ = pi/2
            self.c_radian = right_cos_cos_cos_rule(
                lhs=self.c_radian, cos_1_arg=self.a_radian, cos_2_arg=self.b_radian, right_angle=self.gamma)

            ##########
            # alpha  #
            ##########
            # sin alpha = sin a . sin β / sin b
            self.alpha = sin_rule_calc_top1(top1=self.alpha, top2=self.beta, down1=self.a_radian, down2=self.b_radian)
            # sin alpha = sin a . sin γ / sin c
            self.alpha = sin_rule_calc_top1(top1=self.alpha, top2=self.gamma, down1=self.a_radian, down2=self.c_radian)
            # cos alpha = -cos β cos γ + sin β sin γ cos a
            self.alpha = cos_rule_calc_angle1(
                angle1=self.alpha, angle2=self.beta, angle3=self.gamma, side1=self.a_radian)
            # tan (alpha/2) = √( sin(s–b) sin(s–c) / sin s sin(s–a) )
            # s = (a+b+c)/2
            self.alpha = tan_rule(side1=self.alpha, angle1=self.a_radian, angle2=self.b_radian, angle3=self.c_radian)
            # cos α = cos a sin β = tan b cot c if γ = pi/2
            self.alpha = right_cos_cos_sin(
                lhs=self.alpha, cos_arg=self.a_radian, sin_arg=self.beta, right_angle=self.gamma)
            self.alpha = right_cos_tan_cot_rule(
                lhs=self.alpha, tan_arg=self.b_radian, cot_arg=self.c_radian, right_angle=self.gamma)

            ##########
            # Beta  #
            ##########
            # sin β = sin b . sin alpha / sin a
            self.beta = sin_rule_calc_top1(top1=self.beta, top2=self.alpha, down1=self.b_radian, down2=self.a_radian)
            # sin β = sin b . sin γ / sin c
            self.beta = sin_rule_calc_top1(top1=self.beta, top2=self.gamma, down1=self.b_radian, down2=self.c_radian)
            # cos β = -cos alpha cos γ + sin alpha sin γ cos b
            self.beta = cos_rule_calc_angle1(
                angle1=self.beta, angle2=self.alpha, angle3=self.gamma, side1=self.b_radian)
            # tan (β/2) = √( sin(s–a) sin(s–c) / sin s sin(s–b) )
            # s = (a+b+c)/2
            self.beta = tan_rule(side1=self.beta, angle1=self.b_radian, angle2=self.a_radian, angle3=self.c_radian)

            ##########
            # gamma  #
            ##########
            # sin γ = sin c . sin alpha / sin c
            self.gamma = sin_rule_calc_top1(top1=self.gamma, top2=self.alpha, down1=self.c_radian, down2=self.a_radian)
            # sin γ = sin c . sin β / sin b
            self.gamma = sin_rule_calc_top1(top1=self.gamma, top2=self.beta, down1=self.c_radian, down2=self.b_radian)
            # cos γ = -cos alpha cos β + sin alpha sin β cos c
            self.gamma = cos_rule_calc_angle1(
                angle1=self.gamma, angle2=self.alpha, angle3=self.beta, side1=self.c_radian)
            # tan (γ/2) = √( sin(s–b) sin(s–a) / sin s sin(s–c) )
            # s = (a+b+c)/2
            self.gamma = tan_rule(side1=self.gamma, angle1=self.c_radian, angle2=self.a_radian, angle3=self.b_radian)

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
