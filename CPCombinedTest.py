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
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

Size = 400
Rank = 100

b0 = .05
eta_ada = .1

lamb = .25

proprtions = np.linspace(.01, 1, num = 5)

eps = 1/(2*len(proprtions))
eta_cpd =10
max_time = 600

X = createTensor(Size,Rank)

A_init = initDecomposition(Size,Rank)

A, B, C = A_init[0], A_init[1], A_init[2]

norm_x = linalg.norm(X)

sketching_rates = [(p, True) for p in proprtions] + [(p, False) for p in proprtions]

A, B, C, NRE_A, weights = decompose(X, Rank, sketching_rates, lamb, eps, eta_cpd, A_init, max_time, b0, eta_ada)
print(NRE_A)
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('Error and Weights for Decomposition')
viridis = cm.get_cmap('viridis', len(proprtions))
ax1.set_title('Normalized Error')

handles = [mpatches.Patch(color=viridis(p), label=f'{p} sketching rate') for p in proprtions]
handles = handles + [mlines.Line2D([], [], color='black', marker='o', label='Gradient'),
mlines.Line2D([], [], color='black', marker='+', label='Newton\'s method')
]
# handles.append()
ax1.legend(handles=handles)
ax1.set_yscale('log')
for t in NRE_A:
    e, grad, s = NRE_A[t]
    m = 'o' if grad else 'x'
    ax1.scatter([t], [e], color=viridis(s), marker=m)

weights_t = list(weights.keys())
ax2.set_yscale('log')
ax2.set_title('Probability Wieghts')
for i in range(0,len(sketching_rates)):
    weight_y = []
    for t in weights_t:
        weight_y.append(weights[t][i])
    m = 'o' if sketching_rates[i][1] else 'x'
    ax2.plot(weights_t,weight_y, color=viridis(sketching_rates[i][0]), marker=m)
ax2.legend(handles=handles)
plt.show(block = False)



sketching_rates = [(p, True) for p in proprtions]

A, B, C, NRE_A, weights = decompose(X, Rank, sketching_rates, lamb, eps, eta_cpd, A_init, max_time, b0, eta_ada)
print(NRE_A)
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('Error and Weights for Decomposition')
viridis = cm.get_cmap('viridis', len(proprtions))
ax1.set_title('Normalized Error')

handles = [mpatches.Patch(color=viridis(p), label=f'{p} sketching rate') for p in proprtions]
handles = handles + [mlines.Line2D([], [], color='black', marker='o', label='Gradient'),
mlines.Line2D([], [], color='black', marker='+', label='Newton\'s method')
]
# handles.append()
ax1.legend(handles=handles)
ax1.set_yscale('log')
for t in NRE_A:
    e, grad, s = NRE_A[t]
    m = 'o' if grad else 'x'
    ax1.scatter([t], [e], color=viridis(s), marker=m)

weights_t = list(weights.keys())
ax2.set_yscale('log')
ax2.set_title('Probability Wieghts')
for i in range(0,len(sketching_rates)):
    weight_y = []
    for t in weights_t:
        weight_y.append(weights[t][i])
    m = 'o' if sketching_rates[i][1] else 'x'
    ax2.plot(weights_t,weight_y, color=viridis(sketching_rates[i][0]), marker=m)
ax2.legend(handles=handles)
plt.show(block = False)



sketching_rates = [(p, False) for p in proprtions]

A, B, C, NRE_A, weights = decompose(X, Rank, sketching_rates, lamb, eps, eta_cpd, A_init, max_time, b0, eta_ada)
print(NRE_A)
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('Error and Weights for Decomposition')
viridis = cm.get_cmap('viridis', len(proprtions))
ax1.set_title('Normalized Error')

handles = [mpatches.Patch(color=viridis(p), label=f'{p} sketching rate') for p in proprtions]
handles = handles + [mlines.Line2D([], [], color='black', marker='o', label='Gradient'),
mlines.Line2D([], [], color='black', marker='+', label='Newton\'s method')
]
# handles.append()
ax1.legend(handles=handles)
ax1.set_yscale('log')
for t in NRE_A:
    e, grad, s = NRE_A[t]
    m = 'o' if sketching_rates[i][1] else 'x'
    ax1.scatter([t], [e], color=viridis(s), marker=m)

weights_t = list(weights.keys())
ax2.set_title('Probability Wieghts')
for i in range(0,len(sketching_rates)):
    weight_y = []
    for t in weights_t:
        weight_y.append(weights[t][i])
    m = 'o' if sketching_rates[i][1] else 'x'
    ax2.plot(weights_t,weight_y, color=viridis(sketching_rates[i][0]), marker=m)
ax2.legend(handles=handles)
plt.show()

