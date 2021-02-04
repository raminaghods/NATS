'''
Code for submitted work:

``Multi Agent Active Search using Realistic Depth-Aware Noise Model'',
submitted to 2020 Conference on Robot Learning
author: anonymous

please do not distribute. The code will become public upon acceptance of the paper.
'''

from BinaryTS import BinaryTS
from Random import Random
from NATS import NATS
from RSI import RSI
from plot_animation import plot_animation
from bayesian_optimization import Bayesian_optimizer
from worker_manager import WorkerManager
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from joblib import Parallel, delayed
import seaborn as sb
from matplotlib.pyplot import cm
from scipy import stats
import tikzplotlib

if __name__ == "__main__":

    savepath = 'results/'
    #creating empty files:
    open('resfile.txt', 'w').close()
    open('posteriorfile.txt', 'w').close()
    open('positionfile.txt', 'w').close()
    open('testresults.txt', 'w').close()
    mu = 1.0 # signal intensity to create nonzero entries of vector beta, this parameter is not used for estimation
    theta2 = 0 # signal variance to create nonzero entries of vector beta
    lmbd = 10 # Laplace hyper parameter lmbd = sqrt(eta) where eta is introduced in the paper
    sigma2 = 0.005 # noise variance on observations
    EMitr = 30 # number of iterations for the Expectation-Maximization estimator
    k_arr = np.array([2]) # sparsity rate
    num_trials = 1 # number of trials
    n1 = 8 # length n1 of matrix beta
    n2 = 8 # length n2 of matrix beta
    n = n1*n2
    T =  12# list on number of measurements T
    err = 0.1 # hyperparameter for RSI algorithm
    alpha = 1 # hyper parameter for LATSI algorithm
    noise_vec = np.append(np.append(np.array(1*[sigma2,sigma2]),
                                         np.repeat(4*sigma2,4)),np.repeat(9*sigma2,6))
    num_agents = np.array([2]) # to plot animation agents can only be 1 or 2
    mode = 'asy' #alternatively 'syn' defines synchronous vs. asynchronous parallelisation. we focus on 'asy' in this paper
    n = n1*n2

    kid = -1
    for k in k_arr:
        kid += 1
        aid = 0
        for agents in num_agents:

            #creating a random beta
            rng = np.random.RandomState(0)
            idx = rng.choice(n,k,replace=False)
            beta = np.zeros((n,1))
            beta[idx,:] = mu

            with open('posteriorfile.txt', 'a+') as f:
                print(' '.join(map(str,np.transpose(idx))),'\n',file=f)

            func_class = NATS(beta, n1, mu, theta2, noise_vec, lmbd, EMitr,agents,0)
            worker_manager = WorkerManager(func_caller=func_class, worker_ids=agents, poll_time=1e-15, trialnum=0)
            options = Namespace(max_num_steps=T, num_init_evals=agents, num_workers=agents, mode=mode, GP=func_class,check_performance=False)
            beta_hats = Bayesian_optimizer(worker_manager, func_class, options).optimize(T)

            plot_animation(n1,n2,k,T,mu,'')

            aid += 1
