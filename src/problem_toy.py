import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
import numpy.typing
import os
import math

def f(x: numpy.typing.ArrayLike):
    '''Evaluating the performance of a single solution.'''
    offset = np.abs(np.random.RandomState(2).random(x.shape[0]) / 2.0)
    return np.linalg.norm(x - offset) + np.random.normal(0, np.sqrt(0.1))


def constraint_check(x: numpy.typing.ArrayLike):
    check_0 = x[0] - x[1]
    check_1 = math.cos(x[1]) - math.sin(x[2])
    check_2 = math.atan(x[2]) - x[3]
    check_3 = x[3]*x[3]*x[3] - x[4]*x[6]
    return (check_0, check_1, check_2, check_3) 


