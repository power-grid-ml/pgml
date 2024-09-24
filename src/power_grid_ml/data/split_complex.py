import keras.src.ops.numpy as np

from enum import Enum

class SplitComplexMode(Enum):
    CARTESIAN = 0
    POLAR = 1
    EXPONENTIAL = 2
