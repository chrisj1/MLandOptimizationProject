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

b0 = .05
eta_ada = .1

lamb = .25

proprtions = np.linspace(.01, 1, num = 5)

eps = 1/(2*len(proprtions))
eta_cpd =.39


X = createTensor(Size,Rank)

A_init = initDecomposition(Size,Rank)

A, B, C = A_init[0], A_init[1], A_init[2]

numberOfFibers = Size**2
FibersSampled = (numberOfFibers * proprtions).astype(int)

norm_x = linalg.norm(X)



decompose(X, Rank, proprtions, lamb, eps, eta_cpd, A_init, 1000, b0, eta_ada)