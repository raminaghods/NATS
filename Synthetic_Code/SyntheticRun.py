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

def trials(func_class,mu,max_capital, num_agents, mode, trl):

    rng = np.random.RandomState(trl)
    idx = rng.choice(n,k,replace=False)
    beta = np.zeros((n,1))
    beta[idx,:] = mu

    with open('posteriorfile.txt', 'a+') as f:
        print(' '.join(map(str,np.transpose(idx))),'\n',file=f)

    func_class.set_beta(beta)
    worker_manager = WorkerManager(func_caller=func_class, worker_ids=num_agents, poll_time=1e-15, trialnum=trl)

    options = Namespace(max_num_steps=max_capital, num_init_evals=num_agents, num_workers=num_agents, mode=mode, GP=func_class,check_performance=False)

    beta_hats = Bayesian_optimizer(worker_manager, func_class, options).optimize(max_capital)

    full_recovery_rate = []
    partial_recovery_rate = []

    for i in range(max_capital):
        beta_hat = beta_hats[i]

        est = (beta_hat>(np.amax(beta_hat)/2))
        real = (beta>0)

        partial_recovery_rate.append(np.sum(est==real)/(n))
        correct = 0.0
        if(np.all(est==real)):
            correct = 1.0
        full_recovery_rate.append(correct)

    return full_recovery_rate



if __name__ == "__main__":

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
    k_arr = np.array([5]) # sparsity rate list
    num_trials = 3 # number of trials
    n1 = 6 # length n1 of matrix beta
    n2 = 6
    n = n1*n2
    T =  4# list on number of measurements T
    err = 0.05 # hyperparameter for RSI algorithm
    alpha = 1 # hyper parameter for LATSI algorithm
    noise_vec = np.append(np.append(np.array(1*[sigma2,sigma2]),
                                         np.repeat(4*sigma2,4)),np.repeat(9*sigma2,6))
    num_agents = np.array([4]) # list on number of agents
    mode = 'asy' #alternatively 'syn' defines synchronous vs. asynchronous parallelisation. we focus on 'asy' in this paper
    n = n1*n2
    x_axis = 'measurements' # options for x axis are 'sparsity','agents','measurements'
    recovery_thr = 0.5 # recovry threshold

    full_recovery_rate = np.zeros((num_agents.shape[0], k_arr.shape[0],T, num_trials,4)) # percentage of results where we fully recover a vector beta
    partial_recovery_rate = np.zeros((num_agents.shape[0], k_arr.shape[0],T, num_trials,4))  # percentage of estimating correct entries

    savepath = 'results/'
    filename = ('k-arr_%s_agents_%s_n1_%d_n2_%d_trials_%d.pkl'%(str(k_arr),str(num_agents),n1,n2,num_trials))

    kid = -1
    for k in k_arr:
        kid += 1
        aid = 0
        for agents in num_agents:

            #initialize:
            res_NATS = np.ones((num_trials))
            res_BinaryTS = np.ones((num_trials))
            res_RSI = np.ones((num_trials))
            res_Random = np.ones((num_trials))



            print('agents: %d k=%d'%(agents,k))
            schftseed = T * (num_trials+1)
            beta_temp = np.zeros((n,1)) # temporary value
            num_jobs = 3#40//agents

            result_NATS = Parallel(n_jobs=num_jobs, prefer='processes')(delayed(trials)\
            (NATS(beta_temp, n1, mu, theta2, noise_vec, lmbd, EMitr,agents, schftseed+T*trl),\
            mu,T, agents, mode, schftseed+T*trl) for trl in range(num_trials))
            res_NATS = np.array(result_NATS)

            result_Random = Parallel(n_jobs=num_jobs, prefer='processes')(delayed(trials)\
            (Random(beta_temp,n1,mu,theta2,noise_vec,EMitr, agents,schftseed+T*trl),mu,T, agents, \
            mode, schftseed+T*trl) for trl in range(num_trials))
            res_Random = np.array(result_Random)
            #
            result_RSI = Parallel(n_jobs=num_jobs, prefer='processes')(delayed(trials)\
            (RSI(beta_temp,n1,mu,theta2,noise_vec,lmbd,EMitr,err,agents,schftseed+T*trl),mu,T, \
            agents, mode, schftseed+T*trl) for trl in range(num_trials))
            res_RSI = np.array(result_RSI)
            #
            result_BinaryTS = Parallel(n_jobs=num_jobs, prefer='processes')(delayed(trials)\
            (BinaryTS(beta_temp, n1, mu, theta2, noise_vec, lmbd, EMitr, agents, schftseed+T*trl),\
            mu,T,agents, mode, schftseed+T*trl) for trl in range(num_trials))
            res_BinaryTS = np.array(result_BinaryTS)


            full_recovery_rate[aid,kid,:,:,0] = np.stack(res_NATS).T#NATS
            full_recovery_rate[aid,kid,:,:,1] = np.stack(res_BinaryTS).T#BinaryTS
            full_recovery_rate[aid,kid,:,:,2] = np.stack(res_RSI).T#RSI
            full_recovery_rate[aid,kid,:,:,3] = np.stack(res_Random).T#Random

            aid += 1
            with open(os.path.join(savepath,'k_%d_g_%d_n1_%d_n2_%d_trials_%d.pkl'%(k,agents,n1,n2,num_trials)),'wb') as f:
                pickle.dump([noise_vec,T,full_recovery_rate],f)

    # plot_animation2(n1,n2,k,T,mu)


    with open(os.path.join(savepath,filename),'wb') as f:
        pickle.dump([noise_vec,T,full_recovery_rate],f)

    print('saved!')



    with open(os.path.join(savepath,filename), 'rb') as f:
        data = pickle.load(f)

    measurements = np.zeros((num_agents.shape[0],k_arr.shape[0],3,4))

    if (x_axis == 'sparsity' or x_axis == 'agents'):
        for gid in range(num_agents.shape[0]):
            for kid in range(k_arr.shape[0]):
                recovery = np.mean(data[2][gid,kid,:,:,0],axis=1)
                measurements[gid,kid,1,0] = np.amin(np.argwhere(recovery>recovery_thr))
                m_std_err = stats.sem(data[2][gid,kid,:,:,0], axis=1)
                measurements[gid,kid,0,0] = np.amin(np.argwhere(recovery+m_std_err>recovery_thr))
                measurements[gid,kid,2,0] = np.amin(np.argwhere(recovery-m_std_err>recovery_thr))

                recovery = np.mean(data[2][gid,kid,:,:,1],axis=1)
                measurements[gid,kid,1,1] = np.amin(np.argwhere(recovery>recovery_thr))
                m_std_err = stats.sem(data[2][gid,kid,:,:,1], axis=1)
                measurements[gid,kid,0,1] = np.amin(np.argwhere(recovery+m_std_err>recovery_thr))
                measurements[gid,kid,2,1] = np.amin(np.argwhere(recovery-m_std_err>recovery_thr))

                recovery = np.mean(data[2][gid,kid,:,:,2],axis=1)
                measurements[gid,kid,1,2] = np.amin(np.argwhere(recovery>recovery_thr))
                m_std_err = stats.sem(data[2][gid,kid,:,:,2], axis=1)
                measurements[gid,kid,0,2] = np.amin(np.argwhere(recovery+m_std_err>recovery_thr))
                measurements[gid,kid,2,2] = np.amin(np.argwhere(recovery-m_std_err>recovery_thr))

                recovery = np.mean(data[2][gid,kid,:,:,3],axis=1)
                measurements[gid,kid,1,3] = np.amin(np.argwhere(recovery>recovery_thr))
                m_std_err = stats.sem(data[2][gid,kid,:,:,3], axis=1)
                measurements[gid,kid,0,3] = np.amin(np.argwhere(recovery+m_std_err>recovery_thr))
                measurements[gid,kid,2,3] = np.amin(np.argwhere(recovery-m_std_err>recovery_thr))

    NATScolor='mediumvioletred'
    Randomcolor='steelblue'
    BinaryTScolor='forestgreen'
    RSIcolor='orange'
    # Pointcolor='saddlebrown'

    marker = ["o","d","s","*","X"]

    if (x_axis == 'sparsity'):
        for gid,g in enumerate(num_agents):
            plt.figure(figsize = (8,6))
            sb.tsplot(time=k_arr,data=measurements[gid,:,1,0], color=NATScolor, condition='NATS', linestyle='solid',marker=marker[0])
            sb.tsplot(time=k_arr,data=measurements[gid,:,1,1], color=BinaryTScolor, condition='BinaryTS', linestyle='dashdot',marker=marker[1])
            sb.tsplot(time=k_arr,data=measurements[gid,:,1,2], color=RSIcolor, condition='RSI', linestyle='dotted',marker=marker[2])
            sb.tsplot(time=k_arr,data=measurements[gid,:,1,3], color=Randomcolor, condition='Random', linestyle='dashed',marker=marker[3])

            plt.fill_between(k_arr, measurements[gid,:,0,0],measurements[gid,:,2,0], color=NATScolor, alpha=0.2)
            plt.fill_between(k_arr, measurements[gid,:,0,1],measurements[gid,:,2,1], color=BinaryTScolor, alpha=0.2)
            plt.fill_between(k_arr, measurements[gid,:,0,2],measurements[gid,:,2,2], color=RSIcolor, alpha=0.2)
            plt.fill_between(k_arr, measurements[gid,:,0,3],measurements[gid,:,2,3], color=Randomcolor, alpha=0.2)

            plt.legend()
            plt.xlabel("sparsity (k)",fontsize = 18)
            plt.ylabel("measurements (T)",fontsize = 18)
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)
            plt.title("recovery rate=%1.2f,g=%d"%(recovery_thr,g), fontsize=18)
            plt.savefig('results/sparsity_measurements_recovery_%1.2f_g_%d_n_%d_trials_%d.pdf'%(recovery_thr,g,n,num_trials))
            plt.show()

            plt.figure(figsize = (8,6))
            sb.tsplot(time=k_arr,data=measurements[gid,:,1,0]/g, color=NATScolor, condition='NATS', linestyle='solid',marker=marker[0])
            sb.tsplot(time=k_arr,data=measurements[gid,:,1,1]/g, color=BinaryTScolor, condition='BinaryTS', linestyle='dashdot',marker=marker[1])
            sb.tsplot(time=k_arr,data=measurements[gid,:,1,2], color=RSIcolor, condition='RSI', linestyle='dotted',marker=marker[2])
            sb.tsplot(time=k_arr,data=measurements[gid,:,1,3]/g, color=Randomcolor, condition='Random', linestyle='dashed',marker=marker[3])

            plt.fill_between(k_arr, measurements[gid,:,0,0]/g,measurements[gid,:,2,0]/g, color=NATScolor, alpha=0.2)
            plt.fill_between(k_arr, measurements[gid,:,0,1]/g,measurements[gid,:,2,1]/g, color=BinaryTScolor, alpha=0.2)
            plt.fill_between(k_arr, measurements[gid,:,0,2],measurements[gid,:,2,2], color=RSIcolor, alpha=0.2)
            plt.fill_between(k_arr, measurements[gid,:,0,3]/g,measurements[gid,:,2,3]/g, color=Randomcolor, alpha=0.2)

            plt.legend()
            plt.xlabel("sparsity (k)",fontsize = 18)
            plt.ylabel("time (T/g)",fontsize = 18)
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)
            plt.title("recovery rate=%1.2f,g=%d"%(recovery_thr,g), fontsize=18)
            plt.savefig('results/sparsity_time_recovery_%1.2f_g_%d_n_%d_trials_%d.pdf'%(recovery_thr,g,n,num_trials))
            plt.show()

    elif (x_axis == 'agents'):
        for kid,k in enumerate(k_arr):
            plt.figure(figsize = (8,6))
            sb.tsplot(time=num_agents,data=measurements[:,kid,1,0], color=NATScolor, condition='NATS', linestyle='solid',marker=marker[0])
            sb.tsplot(time=num_agents,data=measurements[:,kid,1,1], color=BinaryTScolor, condition='BinaryTS', linestyle='dashdot',marker=marker[1])
            sb.tsplot(time=num_agents,data=measurements[:,kid,1,2], color=RSIcolor, condition='RSI', linestyle='dotted',marker=marker[2])
            sb.tsplot(time=num_agents,data=measurements[:,kid,1,3], color=Randomcolor, condition='Random', linestyle='dashed',marker=marker[3])

            plt.fill_between(num_agents, measurements[:,kid,0,0],measurements[:,kid,2,0], color=NATScolor, alpha=0.2)
            plt.fill_between(num_agents, measurements[:,kid,0,1],measurements[:,kid,2,1], color=BinaryTScolor, alpha=0.2)
            plt.fill_between(num_agents, measurements[:,kid,0,2],measurements[:,kid,2,2], color=RSIcolor, alpha=0.2)
            plt.fill_between(num_agents, measurements[:,kid,0,3],measurements[:,kid,2,3], color=Randomcolor, alpha=0.2)

            plt.legend()
            plt.xlabel("agents (g)",fontsize = 18)
            plt.ylabel("measurements (T)",fontsize = 18)
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)
            plt.title("recovery rate=%1.2f,k=%d"%(recovery_thr,k), fontsize=18)
            plt.savefig('results/agents_measurements_recovery_%1.2f_k_%d_n_%d_trials_%d.pdf'%(recovery_thr,k,n,num_trials))
            plt.show()

            plt.figure(figsize = (8,6))
            sb.tsplot(time=num_agents,data=measurements[:,kid,1,0]/num_agents, color=NATScolor, condition='NATS', linestyle='solid',marker=marker[0])
            sb.tsplot(time=num_agents,data=measurements[:,kid,1,1]/num_agents, color=BinaryTScolor, condition='BinaryTS', linestyle='dashdot',marker=marker[1])
            sb.tsplot(time=num_agents,data=measurements[:,kid,1,2], color=RSIcolor, condition='RSI', linestyle='dotted',marker=marker[2])
            sb.tsplot(time=num_agents,data=measurements[:,kid,1,3]/num_agents, color=Randomcolor, condition='Random', linestyle='dashed',marker=marker[3])

            plt.fill_between(num_agents, measurements[:,kid,0,0]/num_agents,measurements[:,kid,2,0]/num_agents, color=NATScolor, alpha=0.2)
            plt.fill_between(num_agents, measurements[:,kid,0,1]/num_agents,measurements[:,kid,2,1]/num_agents, color=BinaryTScolor, alpha=0.2)
            plt.fill_between(num_agents, measurements[:,kid,0,2],measurements[:,kid,2,2], color=RSIcolor, alpha=0.2)
            plt.fill_between(num_agents, measurements[:,kid,0,3]/num_agents,measurements[:,kid,2,3]/num_agents, color=Randomcolor, alpha=0.2)

            plt.legend()
            plt.xlabel("agents (g)",fontsize = 18)
            plt.ylabel("time (T/g)",fontsize = 18)
            plt.xticks(fontsize = 18)
            plt.yticks(fontsize = 18)
            plt.title("recovery rate=%1.2f,k=%d"%(recovery_thr,k), fontsize=18)
            plt.savefig('results/agents_time_recovery_%1.2f_k_%d_n_%d_trials_%d.pdf'%(recovery_thr,k,n,num_trials))
            plt.show()

    elif(x_axis == 'measurements'):
        for kid,k in enumerate(k_arr):
            for gid,g in enumerate(num_agents):
                recovery_NATS = np.mean(data[2][gid,kid,:,:,0], axis=1)
                recovery_BinaryTS = np.mean(data[2][gid,kid,:,:,1], axis=1)
                recovery_RSI = np.mean(data[2][gid,kid,:,:,2], axis=1)
                recovery_Random = np.mean(data[2][gid,kid,:,:,3], axis=1)
                f_std_err_NATS = stats.sem(data[2][gid,kid,:,:,0], axis=1)
                f_std_err_BinaryTS = stats.sem(data[2][gid,kid,:,:,1], axis=1)
                f_std_err_RSI = stats.sem(data[2][gid,kid,:,:,2], axis=1)
                f_std_err_Random = stats.sem(data[2][gid,kid,:,:,3], axis=1)

                plt.figure(figsize = (8,6))
                sb.tsplot(time=np.arange(1,T+1),data=recovery_NATS, color=NATScolor, condition='NATS', linestyle='solid',marker=marker[0])
                sb.tsplot(time=np.arange(1,T+1),data=recovery_BinaryTS, color=BinaryTScolor, condition='BinaryTS', linestyle='dashdot',marker=marker[1])
                sb.tsplot(time=np.arange(1,T+1),data=recovery_RSI, color=RSIcolor, condition='RSI', linestyle='dotted',marker=marker[2])
                sb.tsplot(time=np.arange(1,T+1),data=recovery_Random, color=Randomcolor, condition='Random', linestyle='dashed',marker=marker[3])

                plt.fill_between(np.arange(1,T+1), recovery_NATS+f_std_err_NATS, recovery_NATS-f_std_err_NATS, color=NATScolor, alpha=0.2)
                plt.fill_between(np.arange(1,T+1), recovery_BinaryTS+f_std_err_BinaryTS,recovery_BinaryTS-f_std_err_BinaryTS, color=BinaryTScolor, alpha=0.2)
                plt.fill_between(np.arange(1,T+1), recovery_RSI+f_std_err_RSI, recovery_RSI-f_std_err_RSI, color=RSIcolor, alpha=0.2)
                plt.fill_between(np.arange(1,T+1), recovery_Random+f_std_err_Random,recovery_Random-f_std_err_Random, color=Randomcolor, alpha=0.2)

                plt.legend()
                plt.xlabel("number of measurements (T)",fontsize = 18)
                plt.ylabel("full recovery rate",fontsize = 18)
                plt.xticks(fontsize = 18)
                plt.yticks(fontsize = 18)
                plt.ylim((0,1))
                plt.title("k=%d,g=%d"%(k,g), fontsize=18)
                plt.savefig('results/measurements_full_recovery_g_%d_k_%d_n_%d_trials_%d.pdf'%(g,k,n,num_trials))
                plt.show()

                plt.figure(figsize = (8,6))
                sb.tsplot(time=np.arange(1,T+1)/g,data=recovery_NATS, color=NATScolor, condition='NATS', linestyle='solid',marker=marker[0])
                sb.tsplot(time=np.arange(1,T+1)/g,data=recovery_BinaryTS, color=BinaryTScolor, condition='BinaryTS', linestyle='dashdot',marker=marker[1])
                sb.tsplot(time=np.arange(1,T+1)/g,data=recovery_RSI, color=RSIcolor, condition='RSI', linestyle='dotted',marker=marker[2])
                sb.tsplot(time=np.arange(1,T+1)/g,data=recovery_Random, color=Randomcolor, condition='Random', linestyle='dashed',marker=marker[3])

                plt.fill_between(np.arange(1,T+1)/g, recovery_NATS+f_std_err_NATS, recovery_NATS-f_std_err_NATS, color=NATScolor, alpha=0.2)
                plt.fill_between(np.arange(1,T+1)/g, recovery_BinaryTS+f_std_err_BinaryTS,recovery_BinaryTS-f_std_err_BinaryTS, color=BinaryTScolor, alpha=0.2)
                plt.fill_between(np.arange(1,T+1)/g, recovery_RSI+f_std_err_RSI, recovery_RSI-f_std_err_RSI, color=RSIcolor, alpha=0.2)
                plt.fill_between(np.arange(1,T+1)/g, recovery_Random+f_std_err_Random,recovery_Random-f_std_err_Random, color=Randomcolor, alpha=0.2)

                plt.legend()
                plt.xlabel("time (T/g)",fontsize = 18)
                plt.ylabel("full recovery rate",fontsize = 18)
                plt.xlim(0,n)
                plt.xticks(fontsize = 18)
                plt.yticks(fontsize = 18)
                plt.ylim((0,1))
                plt.title("k=%d,g=%d"%(k,g), fontsize=18)
                plt.savefig('results/Time_full_recovery_g_%d_k_%d_n_%d_trials_%d.pdf'%(g,k,n,num_trials))
                plt.show()
