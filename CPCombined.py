import numpy as np
from numpy.linalg import norm
from PerformanceMetrics import MSE
import math
import time
from itertools import permutations
import random
import tensorly as tl
import tensorly.tenalg as tl_alg
from numpy.linalg import pinv, norm
from Utils import error,lookup, proxr, sample_fibers, sampled_kr

def sketching_weight(sketching_rate, weights):
	r = random.uniform(0, sum(weights))
	total_sum = 0

	for i, w in enumerate(weights):
		total_sum += w
		if total_sum > r:
			return sketching_rate[i] if i < len(sketching_rate) else 0

	return 0


def sketch_indices(s, total_col):
	return np.random.choice(
		range(total_col), size=int(s * total_col), replace=False, p=None
	)


def update_factors(A, B, C, X_unfold, Id, lamb, s, rank):
	# Update A
	dim_1, dim_2 = X_unfold[0].shape
	idx = sketch_indices(s, dim_2)
	M = (tl_alg.khatri_rao([A, B, C], skip_matrix=0).T)[:, idx]
	A = (lamb * A + X_unfold[0][:, idx] @ M.T) @ pinv(M @ M.T + lamb * Id)

	# Update B
	dim_1, dim_2 = X_unfold[0].shape
	idx = sketch_indices(s, dim_2)
	M = (tl_alg.khatri_rao([A, B, C], skip_matrix=1).T)[:, idx]
	B = (lamb * B + X_unfold[1][:, idx] @ M.T) @ pinv(M @ M.T + lamb * Id)

	# Update C
	dim_1, dim_2 = X_unfold[1].shape
	idx = sketch_indices(s, dim_2)
	M = (tl_alg.khatri_rao([A, B, C], skip_matrix=2).T)[:, idx]
	C = (lamb * C + X_unfold[2][:, idx] @ M.T) @ pinv(M @ M.T + lamb * Id)

	return A, B, C


def AdaIteration(X, X_unfold,A_mat, B_mat, C_mat, b0, F, errors, n_mb, norm_x,start,sampleErrors):
	dim_vec = X.shape
	dim = len(X.shape)

	A = [A_mat,B_mat,C_mat]
	mu = 0
	print("Gradient!: {}".format(sampleErrors))
	for i in range(10):
		# randomly permute the dimensions
		block_vec = np.random.permutation(dim)
		d_update = block_vec[0]

		# sampling fibers and forming the X_[d] = H_[d] A_[d]^t least squares
		[tensor_idx, factor_idx] = sample_fibers(n_mb, dim_vec, d_update)
		tensor_idx = tensor_idx.astype(int)

		cols = [tensor_idx[:, x] for x in range(len(tensor_idx[0]))]
		X_sample = X[tuple(cols)]
		X_sample = X_sample.reshape((int(X_sample.size / dim_vec[d_update]), dim_vec[d_update]))

		# perform a sampled khatrirao product
		A_unsel = []
		for i in range(d_update):
			A_unsel.append(A[i])
		for i in range(d_update + 1, dim):
			A_unsel.append(A[i])
		H = np.array(sampled_kr(A_unsel, factor_idx))

		# compute the gradient
		g = (1 / n_mb) * (
			A[d_update] @ (H.transpose() @ H + mu * np.eye(F)) - X_sample.transpose() @ H - mu * A[d_update]
		)
		if sampleErrors:
			
			t = time.time()
			e = error(X_unfold[0], norm_x, A[0], A[1], A[2])
			errors[t - start] = e

	return A[0], A[1], A[2], errors

def update_weights(
	X, A, B, C, X_unfold, Id, norm_x, lamb, weights, sketching_rates, rank, nu, eps, b0, F, n_mb
):
	t_sum = 0
	old_error = error(X_unfold[0], norm_x, A, B, C)
	print(weights)
	for i, w in enumerate(weights):
		start = time.time()
		if i == len(weights)-1:
			A_new,B_new,C_new, _ = AdaIteration(X, X_unfold, A, B, C, b0, F, {}, n_mb, norm_x, None, False)
		else:
			s = sketching_rates[i]
			A_new, B_new, C_new = update_factors(A, B, C, X_unfold, Id, lamb, s, rank)
		total_time = time.time() - start
		weights[i] *= np.exp(
			-nu
			/ eps
			* (
				error(X_unfold[0], norm_x, A_new, B_new, C_new)
				- old_error
			)
			/ (total_time)
		)

	weights /= np.sum(weights)
	return

def decompose(X, F, sketching_rates, lamb, eps, nu, Hinit, max_time, b0, n_mb, sample_interval=.5):
	weights = np.array([1] * (len(sketching_rates)+1)) / (len(sketching_rates)+1)

	dim_1, dim_2, dim_3 = X.shape
	A, B, C = Hinit[0], Hinit[1], Hinit[2]

	X_unfold = [tl.unfold(X, m) for m in range(3)]

	norm_x = norm(X)
	I = np.eye(F)

	PP = tl.kruskal_to_tensor((np.ones(F), [A,B,C]))
	e = np.linalg.norm(X - PP) ** 2/norm_x

	NRE_A = {0:e}

	start = time.time()

	sketching_rates_selected = {}
	now = time.time()
	itr = 1
	print(sketching_rates)
	while now - start < max_time:
		s = sketching_weight(sketching_rates, weights)
		print(s)
		if s != 0:
			# Solve Ridge Regression for A,B,C
			A, B, C = update_factors(A, B, C, X_unfold, I, lamb, s, F)
		else:
			A,B,C, NRE_A = AdaIteration(X, X_unfold, A, B, C, b0, F, NRE_A, n_mb, norm_x,start, True)

		# Update weights
		p = np.random.binomial(n=1, p=eps)
		if p == 1 and len(sketching_rates) > 1:
			update_weights(
				X, A, B, C, X_unfold, I, norm_x, lamb, weights, sketching_rates, F, nu, eps, b0, F, n_mb
			)
		now = time.time()            
		e = error(X_unfold[0], norm_x, A, B, C)
		elapsed = now - start
		NRE_A[elapsed] = e
		sketching_rates_selected[elapsed] = s
		print("iteration: {}  t: {}  s: {}   error: {}  rates: {}".format(itr,elapsed, s,e, sketching_rates))
		itr+=1
	return A, B, C, NRE_A,sketching_rates_selected