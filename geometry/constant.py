import numpy as np
from geometry.UfloatOP import UfloatOP

PI = np.pi
EARTH_RADIUS = UfloatOP.from_minmax(6_378_137.000, 6_356_752.314)
SUN_EARTH_DISTANCE = UfloatOP.from_minmax(1.47098074 * 10 ** 11, 1.52097701 * 10 ** 11)

