'''
Code for submitted work:

``Multi Agent Active Search using Realistic Depth-Aware Noise Model'',
submitted to 2020 Conference on Robot Learning
author: anonymous

please do not distribute. The code will become public upon acceptance of the paper.
'''

import numpy as np
import math
import pickle as pkl
import scipy.stats as ss
import scipy.special as ssp
import copy
from scipy.stats import invgauss
import time
from sklearn.metrics import zero_one_loss


class BinaryTS(object):

    def __init__(self, beta, n1, mu, theta2, noise_vec, lmbd, EMitr, n_agents, trl):
        self.mu = mu
        self.lmbd = lmbd
        self.EMitr = EMitr
        self.trl = trl
        self.n_agents = n_agents
        self.gamma = lmbd**2

        self.n = beta.shape[0]
        self.n1 = n1
        self.n2 = int(self.n/self.n1)
        self.L = int(self.n)
        self.M = int(self.n / self.L)

        self.rng = np.random.RandomState(trl)
        self.trl = trl
#        self.err = err

        self.noise_vec = noise_vec

        # self.gamma = self.rng.rand(self.M)
        # self.B = self.rng.rand(self.L,self.L)
        # self.Sig0 = np.kron(np.diag(self.gamma),self.B)
#
#        print('init TS')

    def set_beta(self,beta):
        self.beta = beta
        self.k = np.count_nonzero(self.beta)

    def sample_from_prior(self, num_samples):

        self.rng = np.random.RandomState(self.trl)
        beta_tilde = np.zeros((num_samples, self.n, 1))

        for idx in range(num_samples):

            beta_tilde[idx] = self.rng.laplace(scale=1/self.lmbd,size=(self.n,1))

        return beta_tilde

    def sample_from_prior_per_worker(self, recv_time):

        self.rng = np.random.RandomState(self.trl+recv_time)
        uniform_id = self.rng.rand(self.n,1)
        beta_tilde = np.zeros((self.n,1))
        beta_tilde[uniform_id < self.k/self.n] = self.mu

        return beta_tilde



    def getPosterior(self, X, Y, pos, recv_time):
        self.rng = np.random.RandomState(self.trl+recv_time)
        # print(recv_time)

        pi_1 = np.ones((self.n,1)) * self.k / self.n
        pi_0 = np.ones((self.n,1)) * (1 - self.k / self.n)
        for i in range(X.shape[0]):
            pi_1[X[i,0]] = np.float32(pi_1[X[i,0]] * ss.norm(self.mu,np.sqrt(Y[i,1])).pdf(Y[i,0]))
            pi_0[X[i,0]] = np.float32(pi_0[X[i,0]] * ss.norm(0,np.sqrt(Y[i,1])).pdf(Y[i,0]))
            pi_1[X[i,0]] = pi_1[X[i,0]]/(pi_1[X[i,0]]+pi_0[X[i,0]])
            pi_0[X[i,0]] = pi_0[X[i,0]]/(pi_1[X[i,0]]+pi_0[X[i,0]])

        beta_hat = np.zeros((self.n,1))
        beta_hat[pi_1 > pi_0] = self.mu

        mu_beta = pi_1 * self.mu
        Sig_beta = np.multiply(mu_beta**2,pi_0)+np.multiply((self.mu-mu_beta)**2,pi_1)

        uniform_id = self.rng.rand(self.n,1)
        beta_tilde = np.zeros((self.n,1))
        beta_tilde[uniform_id < pi_1] = self.mu


        return beta_tilde,pi_0,pi_1,beta_hat,Sig_beta#pi_0,beta_rsi

    def ActiveSearch(self,points_dict,qinfo):#beta,mu,theta2,sigma2,lmbd,EMitr,T,trl=np.random.randint(1000),X,Y):

        recv_time = qinfo.send_time
        self.rng = np.random.RandomState(self.trl+recv_time)
        wid =  qinfo.worker_id
        if qinfo.compute_posterior:
            X = points_dict['X']
            Y = points_dict['Y']
#            print('worker #',wid,'writes up to sensing step',X.shape[0])
#            print('worker #',wid,'reads  up to sensing step',X.shape[0])
            [_,pos] = points_dict['par']
            beta_tilde,pi_0,pi_1,beta_hat,Sig_beta = self.getPosterior(X,Y,pos,recv_time)
        else:#this branch True on initial evaluations for agents with samples from prior
            pi_1 = np.ones((self.n,1)) * self.k / self.n
            pi_0 = np.ones((self.n,1)) * (1 - self.k / self.n)
            beta_tilde = self.sample_from_prior_per_worker(recv_time)
            beta_hat = np.zeros((self.n,1))
            beta_hat[pi_1 > pi_0] = self.mu
            mu_beta = pi_1 * self.mu
            Sig_beta = np.multiply(mu_beta**2,pi_0)+np.multiply((self.mu-mu_beta)**2,pi_1)
            pos = np.zeros((self.n_agents))


        max_reward = -math.inf


        k = np.count_nonzero(self.beta)
        beta_bar = np.zeros((self.n,1)) + self.mu * k / self.n
#
        oldl = pos[wid] % self.n1
        oldh = (pos[wid]-oldl)/self.n1
        dist = np.zeros((self.n2,self.n1,4))+1e10
        loss = np.zeros((self.n2,self.n1,4))+1e10
        avg_loss = 0.001
        avg_dist = 0
        count = 0
        for h in range(0,self.n2):
            for l in range(0,self.n1):
                for d in range(0,4):
                    if(h==0 and d==2):
                        continue
                    elif(l==0 and d==3):
                        continue
                    elif(l==self.n1-1 and d==1):
                        continue
                    elif(h==self.n2-1 and d==0):
                        continue
                    x,nonzero_idx,noise_var = self.create_directional_sensor(l,h,d)

                    prob_err = np.zeros((self.n,1))
                    for j in range(self.n):
                        if(beta_tilde[j]>0 and beta_hat[j]==0):
                            prob_err[j] = 1
                        if(beta_tilde[j]==0 and beta_hat[j]>0):
                            prob_err[j] = 1
                    for i in range(x.shape[0]):
                        if(pi_0[x[i,0]] == 0):
                            theta = 0
                        elif(pi_1[x[i,0]] == 0):
                            theta = self.mu
                        else:
                            theta = self.mu/2 - (noise_var[i]/self.mu) * np.log(pi_1[x[i,0]]/pi_0[x[i,0]])
                        if(beta_tilde[x[i,0]]>0.0):
                            prob_err[x[i,0]] = ssp.erfc((self.mu-theta)/(np.sqrt(2*noise_var[i])))
                        else:
                            prob_err[x[i,0]] = ssp.erfc((theta)/(np.sqrt(2*noise_var[i])))


                    loss[h,l,d] = np.sum(prob_err)
                    # loss[h,l,d] = np.linalg.norm(prob_err,2)
                    dist[h,l,d] = (np.absolute(oldl-l)**2+np.absolute(oldh-h)**2)

                    avg_loss += loss[h,l,d]
                    avg_dist += dist[h,l,d]
                    count += 1


        avg_loss /= count
        avg_dist /= count

        reward = - loss#/avg_loss - dist/avg_dist
        [hm,lm,dm] = np.unravel_index(reward.argmax(), reward.shape)
        bestx,bestidx,best_noise_var = self.create_directional_sensor(lm,hm,dm)
        pos[wid] = float(hm*self.n1+lm)

        with open('resfile.txt', 'a+') as f:
            print(wid,'\t',' '.join(map(str, np.squeeze(np.round(beta_tilde,2)))),'\n',file=f)
            print(wid,'\t',' '.join(map(str,bestidx)),'\n',file=f)

        with open('posteriorfile.txt', 'a+') as f:
            print(wid,'\t',' '.join(map(str, np.squeeze(np.round(beta_hat,2)))),'\n',file=f)
            print(wid,'\t',' '.join(map(str, np.squeeze(np.round(Sig_beta,2)))),'\n',file=f)

        with open('positionfile.txt', 'a+') as f:
            print(wid,'\t',float(pos[wid]),'\n',file=f)


        # # %% take a new observation
        # one-sided noise:
        epsilon = np.abs(np.multiply(self.rng.randn(bestx.shape[0],1),np.reshape(np.sqrt(best_noise_var),(-1,1))))
        epsilon_neg = -1*np.abs(np.multiply(self.rng.randn(bestx.shape[0],1),np.reshape(np.sqrt(best_noise_var),(-1,1))))
        obs = np.zeros((bestx.shape[0],1))
        for i in range(bestx.shape[0]):
            obs[i] = self.beta[bestx[i,0]]
        epsilon[obs>0] = epsilon_neg[obs>0]
        y_temp = np.squeeze(obs+epsilon)
        y_temp[y_temp<0.0] = 0.0
        y_temp[y_temp>self.mu] = self.mu
        y = np.zeros((bestx.shape[0],2))
        y[:,0] = y_temp
        y[:,1] = best_noise_var


        if not qinfo.compute_posterior:
            result = {'x':[bestx], 'y':[y], 'par':[beta_hat,pos], 'pre-eval':True}
        else:
            result = {'x':bestx,'y':y,'par':[beta_hat,pos]}

        with open(qinfo.result_file, 'wb') as f:
            pkl.dump(result, f)

        print('BinaryTS. trial: ',self.trl,' worker: ',wid,'recv_time: ',recv_time,'finished writing')

        return result


    def create_directional_sensor(self,l,h,d):
#        l = i % n1
#        h = (i-l)/n1
        if(d == 0):
            ls = np.array([l-1,l,l-2,l-1,l,l+1,l-3,l-2,l-1,l,l+1,l+2])
            hs = np.array([h+1,h+1,h+2,h+2,h+2,h+2,h+3,h+3,h+3,h+3,h+3,h+3])
#            non_zero_idx_t = np.array([i+n1-1,i+n1,i+2*n1,i+2*n1-1,i+2*n1-2,
#                                     i+2*n1+1,i+3*n1-3,i+3*n1-2,i+3*n1-1,
#                                     i+3*n1,i+3*n1+1,i+3*n1+2])
        elif(d == 1):
            ls = np.array([l+1,l+1,l+2,l+2,l+2,l+2,l+3,l+3,l+3,l+3,l+3,l+3])
            hs = np.array([h+1,h,h+2,h+1,h,h-1,h+3,h+2,h+1,h,h-1,h-2])
#            non_zero_idx_t = np.array([i+1,i+n1+1,i+2,i+2-n1,i+2+n1,i+2+2*n1,
#                                     i+3,i+3-n1,i+3-2*n1,i+3+n1,i+3+2*n1,i+3+3*n1])
        elif(d == 2):
            ls = np.array([l,l+1,l-1,l,l+1,l+2,l-2,l-1,l,l+1,l+2,l+3])
            hs = np.array([h-1,h-1,h-2,h-2,h-2,h-2,h-3,h-3,h-3,h-3,h-3,h-3])
#            non_zero_idx_t = np.array([i-n1,i-n1+1,i-2*n1,i-2*n1-1,i-2*n1+1,i-2*n1+2,
#                                     i-3*n1,i-3*n1-1,i-3*n1-2,i-3*n1+1,i-3*n1+2,i-3*n1+3])
        elif(d == 3):
            ls = np.array([l-1,l-1,l-2,l-2,l-2,l-2,l-3,l-3,l-3,l-3,l-3,l-3])
            hs = np.array([h-1,h,h-2,h-1,h,h+1,h-3,h-2,h-1,h,h+1,h+2])
#            non_zero_idx_t = np.array([i-1,i-1-n1,i-2,i-2-n1,i-2-2*n1,i-2+n1,
#                                     i-3,i-3-n1,i-3-2*n1,i-3-3*n1,i-3+n1,i-3+2*n1])
        else:
            print('wrong d parameter')

        non_zero_idx = []
        noise_var = []
        count = 0
        for ii in range(0,12):
            if(ls[ii]<self.n1 and ls[ii]>=0 and hs[ii]<self.n2 and hs[ii]>=0):
                pos = int(hs[ii]*self.n1+ls[ii])
                non_zero_idx.append(pos)
                noise_var.append(self.noise_vec[ii])
                count = count+1
        # x = np.zeros((count,self.n))
        # for jj in range(count):
        #     x[jj,non_zero_idx[jj]] = 1

        x = np.zeros((count,1),dtype=int)
        for jj in range(count):
            x[jj,:] = non_zero_idx[jj]
        return x,np.array(non_zero_idx),np.array(noise_var)
