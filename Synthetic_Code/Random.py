'''
Code for the work:

``Multi Agent Active Search using Realistic Depth-Aware Noise Model'', Ramina Ghods, William J Durkin and Jeff Schneider

(C) Ramina Ghods 2020 (rghods@cs.cmu.edu)
Please cite the following paper to use the code:

@article{ghods2020multi,
  title={Multi-Agent Active Search using Realistic Depth-Aware Noise Model},
  author={Ghods, Ramina and Durkin, William J and Schneider, Jeff},
  journal={arXiv preprint arXiv:2011.04825},
  year={2020}
}
'''


import numpy as np
import math
import pickle as pkl
import scipy.stats as ss
import copy
from scipy.stats import invgauss
import time


class Random(object):

    def __init__(self, beta, n1, mu, theta2, noise_vec, EMitr,n_agents, trl):
        self.mu = mu
        self.trl = trl

        self.n = beta.shape[0]
        self.n1 = n1
        self.n2 = int(self.n/self.n1)
        self.EMitr = EMitr
        self.n_agents = n_agents
        self.rng = np.random.RandomState(trl)

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

            beta_tilde[idx] = self.rng.laplace(scale=1,size=(self.n,1))

        return beta_tilde



    def getPosterior(self, X, Y, L, recv_time):

        a = 0.1
        b = 1
        self.rng = np.random.RandomState(self.trl+recv_time)
        # print(recv_time)
        XT_X = np.matmul(np.transpose(X),X)
        XT_Y = np.matmul(np.transpose(X),np.reshape(Y[:,0],(-1,1)))
        k = np.count_nonzero(self.beta)
        beta_bar = np.zeros((self.n,1)) #+ self.mu * k / self.n
        gamma = np.zeros((self.n))
        beta_hat = np.zeros((self.n,1))
        # Sig_beta = np.zeros((self.n,self.n))+1e8
        idx = []
        for j in range(self.EMitr):
            alpha = np.zeros((self.n,1))+1e-8
            A = np.diag(np.squeeze(alpha))
            Sig_beta = np.linalg.inv(np.matmul(np.matmul(np.transpose(X),np.diag(1./np.squeeze(Y[:,1]))),X)+A)
            beta_hat_tmp = np.matmul(np.matmul(Sig_beta,np.transpose(X)),np.matmul(np.diag(1./np.squeeze(Y[:,1])),
                                           np.reshape(Y[:,0],(-1,1))-np.matmul(X,beta_bar)))+beta_bar

            for ii in range(self.n):
                alpha[ii,0] = (1+2*a)/(2*b+beta_hat_tmp[ii,0]**2+Sig_beta[ii,ii])
                if(ii not in idx):
                    beta_hat[ii,0] = beta_hat_tmp[ii]
                if(alpha[ii,0]>1e8):
                    alpha[ii,0] = 1e8
                    beta_hat[ii,0] = 0.0
                    idx.append(ii)

        return None,None,None,beta_hat,None

    def ActiveSearch(self,points_dict,qinfo):#beta,mu,theta2,sigma2,lmbd,EMitr,T,trl=np.random.randint(1000),X,Y):

        recv_time = qinfo.send_time
        self.rng = np.random.RandomState(self.trl+recv_time)
        wid =  qinfo.worker_id

        if qinfo.compute_posterior:
            X = points_dict['X']
            Y = points_dict['Y']
            [_,pos] = points_dict['par']
            _,_,_,beta_hat,_ = self.getPosterior(X, Y, _, recv_time)

        else:
            pos = np.zeros((self.n_agents))
            beta_hat = np.zeros((self.n,1))

        h = self.rng.randint(0,self.n2)
        l = self.rng.randint(0,self.n1)
        d = self.rng.randint(0,4)
        if(h==0 and d==2):
            d = 0
        elif(l==0 and d==3):
            d = 1
        elif(l==self.n1-1 and d==1):
            d = 3
        elif(h==self.n2-1 and d==0):
            d = 2
        bestx,bestidx,best_noise_var = self.create_directional_sensor(l,h,d)
        pos[wid] = float(h*self.n1+l)




        #%% take a new observation
        # one-sided noise:
        epsilon = np.abs(np.multiply(self.rng.randn(bestx.shape[0],1),np.reshape(np.sqrt(best_noise_var),(-1,1))))
        epsilon_neg = -1*np.abs(np.multiply(self.rng.randn(bestx.shape[0],1),np.reshape(np.sqrt(best_noise_var),(-1,1))))
        obs = np.matmul(bestx,self.beta)
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


        print('Random. trial: ',self.trl,' worker: ',wid,'recv_time: ',recv_time,'finished writing')

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
        x = np.zeros((count,self.n))
        for jj in range(count):
            x[jj,non_zero_idx[jj]] = 1
#        print(i,d)
#        print(np.array(non_zero_idx))

        return x,np.array(non_zero_idx),np.array(noise_var)
