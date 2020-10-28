import numpy as np
import tensorly as tl
import tensorly.tenalg as tl_alg
import numpy.matlib as matlib
import time
import pickle

from numpy import r_

import cv2

from cv2 import VideoWriter, VideoWriter_fourcc

def initDecomposition(Size, F):
    # initialize the latent factors
    Hinit = []
    for d in range(3):
        Hinit.append(np.random.random((Size, F)))
    return Hinit

def createTensor(size,F):
    X = np.zeros((size, size, size))

    # generate true latent factors
    A = []
    for i in range(3):
        A.append(np.random.random((size, F)))

    A_gt = A

    # form the tensor
    for k in range(size):
        X[:, :, k] = A[0] @ np.diag(A[2][k, :]) @ np.transpose(A[1])
    return X

def sample_fibers(n_fibers, dim_vec, d):
    dim_len = len(dim_vec)
    tensor_idx = np.zeros((n_fibers, dim_len))

    # randomly select fibers for dimensions not d
    for i in r_[0 : d, d + 1 : dim_len]:
        tensor_idx[:, i] = np.random.random_integers(
            0, high=dim_vec[i] - 1, size=(n_fibers)
        )

    factor_idx = tensor_idx[:, r_[0 : d, d + 1 : dim_len]]

    tensor_idx = tl.tenalg.kronecker([tensor_idx, np.ones((dim_vec[d], 1))])

    tensor_idx[:, d] = (
        matlib.repmat(np.arange(0, dim_vec[d]).transpose(), n_fibers, 1)
    ).reshape(tensor_idx[:, d].shape)

    return (tensor_idx, factor_idx)


def sampled_kr(A_unsel, factor_idx):
    l = [x for x in range(len(A_unsel) - 1, -1, -1)]
    H = [A_unsel[l[0]][int(x), :] for x in factor_idx[:, l[0]]]
    for i in l[1:]:
        H = np.multiply(H, np.array([A_unsel[i][int(x), :] for x in factor_idx[:, i]]))
        H = np.array(H)
    return H


def lookup(x, idx):
    ret = list(x)
    for i in idx:
        ret = ret[int(i)]
    return np.array(ret)

def proxr(Hb, d):
    return Hb

def error(X0, norm_x, A, B, C):
    X_bar = A @ (tl_alg.khatri_rao([A, B, C], skip_matrix=0).T)
    return tl.norm(X0 - X_bar)/norm_x

def tensorToVideo(X, name):
    X = X * 256

    width, height, frames = X.shape
    FPS = 30

    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(name + '.avi', fourcc, float(FPS), (width, height))

    for i in range(frames):
        frame = np.uint8(np.kron(X[...,i], np.array([1,1,1])))
        video.write(frame)
    video.release()

def save_trial_data(algo, X, GT, timing, MSE, NRE, A, params):
    t = time.time()
    name = "data/" + algo + "-" + str(t) + ".dat"
    data = {
        "X": X,
        "GT": GT,
        "timing": timing,
        "MSE": MSE,
        "NRE": NRE,
        "A": A,
        "params": params,
    }
    with open(name, "wb") as f:
        pickle.dump(data, f)

def synteticTensor(I, F):
    X = np.zeros((I[0], I[1], I[2]))

    # generate true latent factors
    A = []
    for i in range(3):
        A.append(np.random.random((I[i], F)))

    A_gt = A

    # form the tensor
    for k in range(I[2]):
        X[:, :, k] = A[0] @ np.diag(A[2][k, :]) @ np.transpose(A[1])

    return X, A_gt

def videoToTensor(filename):
    cap = cv2.VideoCapture(filename)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    X = np.empty((frameCount, frameHeight, frameWidth))

    fc = 0
    ret = True

    while (fc < frameCount):
        ret, frame = cap.read()
        X[fc] = frame[:, :, 0]
        fc += 1

    cap.release()
    s = int(min(frameCount, frameHeight, frameWidth))
    print((frameCount, frameHeight, frameWidth))

    X = X[:s, :s, :s]
    return X