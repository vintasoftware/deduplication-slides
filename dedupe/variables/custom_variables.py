import logging

from dedupe.variables.latlong import LatLongType
from dedupe.variables.string import ShortStringType
from recordlinkage.algorithms.distance import _haversine_distance
from recordlinkage.algorithms.numeric import _exp_sim
import jellyfish
import numpy as np


logger = logging.getLogger(__name__)


class JaroWinklerType(ShortStringType):
    type = "JaroWinkler"

    def __init__(self, definition):
        super().__init__(definition)

        self.comparator = jellyfish.jaro_winkler


class ExpLatLongType(LatLongType):
    type = 'ExpLatLong'

    @staticmethod
    def comparator(x, y):
        dist = _haversine_distance(*[*x, *y])
        return _exp_sim(
            np.float32(dist),
            scale=np.float32(0.1),
            offset=np.float32(0.01))
