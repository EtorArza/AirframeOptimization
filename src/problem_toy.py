import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
import numpy.typing
import os


def f(x: numpy.typing.ArrayLike):
    '''Evaluating the performance of a single solution.'''
    offset = np.random.RandomState(2).random(x.shape[0]) / 2.0
    return np.linalg.norm(x - offset)


def constraint_check(x: numpy.typing.ArrayLike):
    
    # constraint 0: x[0] > x[1] > x[2] > x[3]
    check_0 = False
    if x[0] > x[1] > x[2] > x[3] > x[4]:
        check_0 = True

    return tuple([check_0])


