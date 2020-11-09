from switch import adaIteration, error, cpdmwuiteration
from Utils import save_trial_data, createTensor, initDecomposition, videoToTensor
import time
import pickle
import os
import numpy as np
import numpy.linalg as linalg
import collections
import matplotlib.pyplot as plt
from math import sqrt, log
import tensorly as tl
from CPCombined import *

Size = 200
Rank = 50

b0 = 1
eta_ada = 1

lamb = .01

proprtions = np.linspace(.01, 1, num = 10)

eps = 1/(len(proprtions))
eta_cpd = sqrt(2 * log(len(proprtions)))


X = createTensor(Size,Rank)

A_init = initDecomposition(Size,Rank)

A, B, C = A_init[0], A_init[1], A_init[2]

numberOfFibers = Size**2
FibersSampled = (numberOfFibers * proprtions).astype(int)

n_mb = 100#200*200//5#FibersSampled[5]
norm_x = linalg.norm(X)



decompose(X, Rank, proprtions, lamb, eps, eta_cpd, A_init, 100, b0, n_mb)