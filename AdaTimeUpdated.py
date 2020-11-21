from AdaCPDTime import AdaCPDTime
from Utils import save_trial_data, createTensor, initDecomposition, videoToTensor
import time
import pickle
import os
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm

fiberPropotions = [.001, .005, .01, .05, .1, .25, .5, 1]

Rank = 100

Sizes = [(40,50,12)]

Trials = 1

arrangements = []

maxtime = 20

dir = str(time.time())
os.mkdir(dir)

for fiberPropotion in fiberPropotions:
    for Size in Sizes:
        for trial in range(Trials):
            arrangements.append((fiberPropotion, Size, trial, Rank, maxtime, dir))

def saveAdaTimeTrial(X, fiberPropotion, Size, trial, Rank,b0, max_time, error,dir):
    filename = '{}/AdaResults_{}_{}_{}_{}_{}_{}.dat'.format(dir,fiberPropotion, Size, trial, Rank,b0, max_time)
    results = {
        'fiberPropotion':fiberPropotion, 'Size':Size, 'trial':trial, 'Rank':Rank, 'b0':b0, 'max_time':max_time, 'error':error
    }
    with open(filename, 'wb') as f:
        pickle.dump(results, f)


def runTest(conf):
    fiberPropotion, Size, trial, Rank, max_time,dir = conf
    print("Running video trial with the following:\n\tProporion of Fibers = {}\n\tSize = {}\n\tRank = {}\n\tTrialNumber = {}\n\tMax Time = {}".format(fiberPropotion, Size, Rank, trial, maxtime))
    numberOfFibers = Size[0]*Size[1]
    FibersSampled = max(int(numberOfFibers * fiberPropotion),1)

    #Create tensor
    # X = videoToTensor('600Test.mp4')
    X = createTensor((40,50,12), Rank)

    # print(X.shape)
    # init starting
    A_init = initDecomposition(Size,Rank)

    b0 = .25

    # print(X, b0, FibersSampled)

    error,_ = AdaCPDTime(X, b0, FibersSampled, max_time, A_init, sample_interval=.5,eta=1)
    saveAdaTimeTrial(X, fiberPropotion, Size, trial, Rank, b0, max_time, error,dir)

num_cores = 2
datas = Parallel(n_jobs=num_cores, verbose=100)(delayed(runTest)(i) for i in tqdm(arrangements, position=0))

