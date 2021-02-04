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
from sklearn.metrics import zero_one_loss
import scipy.special as ssp


class NATS(object):

    def __init__(self, beta, n1, mu, theta2, noise_vec, lmbd, EMitr, n_agents, trl):
        self.mu = mu
        self.beta = beta
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

    def set_beta(self,beta):
        self.beta = beta

    def sample_from_prior(self, num_samples):

        self.rng = np.random.RandomState(self.trl)
        beta_tilde = np.zeros((num_samples, self.n, 1))

        for idx in range(num_samples):

            beta_tilde[idx] = self.rng.laplace(scale=1/self.lmbd,size=(self.n,1))

        return beta_tilde

    def sample_from_prior_per_worker(self, recv_time):

        self.rng = np.random.RandomState(self.trl+recv_time)

        beta_tilde = np.maximum(self.rng.laplace(scale=1/self.lmbd,size=(self.n,1)),np.zeros((self.n,1)))


        return beta_tilde



    def getPosterior(self, X, Y, pos, recv_time):
        a = 0.1
        b = 1
        self.rng = np.random.RandomState(self.trl+recv_time)
        # print(recv_time)
        XT_X = np.matmul(np.transpose(X),X)
        XT_Y = np.matmul(np.transpose(X),np.reshape(Y[:,0],(-1,1)))
        k = np.count_nonzero(self.beta)
        beta_bar = np.zeros((self.n,1))
        gamma = np.zeros((self.n))
        beta_hat = np.zeros((self.n,1))
        # Sig_beta = np.zeros((self.n,self.n))+1e8
        idx = []
        gamma = np.zeros((self.n,1))+1e+8
        for j in range(self.EMitr):
            Gamma = np.diag(np.squeeze(gamma))
            Sig_beta = np.linalg.inv(np.matmul(np.matmul(np.transpose(X),np.diag(1./np.squeeze(Y[:,1]))),X)+np.linalg.inv(Gamma))
            beta_hat_tmp = np.matmul(np.matmul(Sig_beta,np.transpose(X)),np.matmul(np.diag(1./np.squeeze(Y[:,1])),
                                           np.reshape(Y[:,0],(-1,1))))

            for ii in range(self.n):
                gamma[ii,0] = (2*b+beta_hat_tmp[ii,0]**2+Sig_beta[ii,ii])/(1+2*a)
                if(ii not in idx):
                    beta_hat[ii,0] = beta_hat_tmp[ii]
                if(gamma[ii,0]<1e-8):
                    gamma[ii,0] = 1e-8
                    beta_hat[ii,0] = 0.0
                    idx.append(ii)

        min_eig = np.amin(np.real(np.linalg.eigvals(Sig_beta)))
        if min_eig < 0:
            Sig_beta -= 1.1* min_eig * np.eye(*Sig_beta.shape)

        beta_tilde = np.maximum(np.reshape(self.rng.multivariate_normal(np.squeeze(beta_hat),Sig_beta),(self.n,1)),np.zeros((self.n,1)))

        floorsample = ((beta_tilde)>(np.amax(beta_tilde)/2))
        beta_tilde = np.zeros((self.n,1))
        beta_tilde[floorsample] = self.mu

        _ = 1

        return beta_tilde,Gamma,Sig_beta,beta_hat,pos#pi_0,beta_rsi

    def gibbs_invgauss(self,itr,XT_X,XT_Y):

        np.random.seed(self.trl+itr-1000)
        self.rng = np.random.RandomState(self.trl+itr-1000)
        tauinv_vec = 1/(np.random.rand(self.n)*self.mu)
        for i in range(itr):
            Sig = np.linalg.inv(XT_X+self.sigma2*np.diag(tauinv_vec)+1e-3*np.eye(self.n))
            beta = self.rng.multivariate_normal(np.squeeze(np.matmul(Sig,XT_Y)),self.sigma2*Sig)
            for j in range(self.n):
                tauinv_vec[j] = invgauss.rvs(np.sqrt(self.sigma2)*(self.lmbd**(1/3))/np.abs(beta[j]))*(self.lmbd**(2/3))
        Sig_beta = self.sigma2*Sig
        beta_hat = np.matmul(Sig,XT_Y)
        beta_tilde = np.maximum(np.reshape(beta,(self.n,1)),np.zeros((self.n,1)))
        return beta_tilde,Sig_beta,beta_hat,tauinv_vec

    def ActiveSearch(self,points_dict,qinfo):#beta,mu,theta2,sigma2,lmbd,EMitr,T,trl=np.random.randint(1000),X,Y):

        recv_time = qinfo.send_time
        self.rng = np.random.RandomState(self.trl+recv_time)
        wid =  qinfo.worker_id
        if qinfo.compute_posterior:
            X = points_dict['X']
            Y = points_dict['Y']
            [_,pos] = points_dict['par']
            beta_tilde,Gamma,Sig_beta,beta_hat,_ = self.getPosterior(X,Y,pos,recv_time)
        else:#this branch True on initial evaluations for agents with samples from prior
            beta_tilde = self.sample_from_prior_per_worker(recv_time)
            beta_hat = np.zeros((self.n,1))
            Sig_beta = np.diag(100*np.ones((self.n)))
            beta_rsi = self.beta
            pos = np.zeros((self.n_agents))


        max_reward = -math.inf
        k = np.count_nonzero(self.beta)
        beta_bar = np.zeros((self.n,1))
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

                    if (qinfo.compute_posterior):
                        Xt = np.append(X,x,axis=0)

                        Sig = np.linalg.inv(np.matmul(np.matmul(np.transpose(Xt),
                                                                     np.diag(1./np.append(np.squeeze(Y[:,1]),noise_var))),Xt)+np.linalg.inv(Gamma))
                        b = np.matmul(np.matmul(Sig,np.transpose(Xt)),
                                             np.diag(1./np.append(np.squeeze(Y[:,1]),noise_var)))


                        b1 = b[:,:-1*x.shape[0]]
                        b2 = b[:,-1*x.shape[0]:]
                        b1Y_b = np.matmul(b1,np.reshape(Y[:,0],(-1,1)))-beta_tilde
                        xTb = np.matmul(x,beta_tilde)

                        # # norm 2 loss:
                        # loss[h,l,d] = (np.linalg.norm(b1Y_b)**2+\
                        #               (np.sum(noise_var)+np.linalg.norm(xTb)**2)*(np.linalg.norm(b2)**2)+\
                        #               np.ndarray.item(2*np.matmul(np.matmul(np.transpose(b1Y_b),b2),xTb)))\
                        #               /(np.linalg.norm(np.matmul(b1,np.reshape(Y[:,0],(-1,1))))**2+\
                        #                             (np.sum(noise_var)+np.linalg.norm(xTb)**2)*(np.linalg.norm(b2)**2)+\
                        #                             np.ndarray.item(2*np.matmul(np.matmul(np.transpose(b1Y_b),b2),xTb)))

                        # loss with beta_expectation:
                        xTb = np.matmul(x,beta_tilde)
                        beta_hat_t = np.matmul(b1,np.reshape(Y[:,0],(-1,1)))+np.matmul(b2,xTb)
                        est_t = (np.round(beta_hat_t)>(np.amax(beta_hat_t)/2))
                        real_t = (np.round(beta_tilde)>(np.amax(beta_tilde)/2))
                        loss[h,l,d] = 1*(np.linalg.norm(beta_hat_t-beta_tilde,1)+0.01*int(~np.all(est_t==real_t)))

                        # # norm 1 loss:
                        # xTb = np.matmul(x,beta_tilde)
                        # Si = np.reshape(noise_var,(-1,1))
                        # E_abs_y = np.sqrt(2/math.pi)*np.multiply(np.sqrt(Si),np.exp(-(xTb**2/(2*Si))))\
                        #     +np.multiply(xTb,ssp.erf(xTb/np.sqrt(2*Si)))
                        # loss[h,l,d] = np.linalg.norm(b1Y_b,1)+np.sum(np.matmul(np.absolute(b2),E_abs_y))

                        dist[h,l,d] = np.sqrt(np.absolute(oldl-l)**2+np.absolute(oldh-h)**2)


                    else:
                        Vinv = np.eye(self.n)*1e3
                        tmp = np.linalg.inv(np.diag(noise_var)+np.matmul(np.matmul(x,Vinv),np.transpose(x)))
                        b = np.matmul(np.matmul(Vinv,np.transpose(x)),tmp)
                        b2 = b
                        b1Y_b = -beta_tilde
#                        #norm 2 loss:
#                        reward = -1*(np.linalg.norm(beta_tilde)**2+
#                                     0.01*(self.sigma2*x.shape[0]+np.linalg.norm(xTb)**2)*(np.linalg.norm(b)**2)-
#                                     np.ndarray.item(2*np.matmul(np.matmul(np.transpose(beta_tilde),b),xTb)))

                        # loss with expectation:
                        xTb = np.matmul(x,beta_tilde)
                        beta_hat_t = np.matmul(b,xTb)-np.matmul(b,np.matmul(x,beta_bar))+beta_bar
                        est_t = (np.round(beta_hat_t)>(np.amax(beta_hat_t)/2))
                        real_t = (np.round(beta_tilde)>(np.amax(beta_tilde)/2))
                        loss[h,l,d] = (np.linalg.norm(beta_hat_t-beta_tilde,1)+0.01*int(~np.all(est_t==real_t)))
                        dist[h,l,d] = 0.001



                    avg_loss += loss[h,l,d]
                    avg_dist += dist[h,l,d]
                    count += 1




        avg_loss /= count
        avg_dist /= count

        reward = - loss/avg_loss - 0.1*dist/avg_dist
        [hm,lm,dm] = np.unravel_index(reward.argmax(), reward.shape)
        bestx,bestidx,best_noise_var = self.create_directional_sensor(lm,hm,dm)
        pos[wid] = float(hm*self.n1+lm)

        with open('resfile.txt', 'a+') as f:
            print(wid,'\t',' '.join(map(str, np.squeeze(np.round(beta_tilde,2)))),'\n',file=f)
            print(wid,'\t',' '.join(map(str,bestidx)),'\n',file=f)

        with open('posteriorfile.txt', 'a+') as f:
            print(wid,'\t',' '.join(map(str, np.squeeze(np.round(beta_hat,2)))),'\n',file=f)
            print(wid,'\t',' '.join(map(str, np.squeeze(np.round(np.diag(Sig_beta),2)))),'\n',file=f)

        with open('positionfile.txt', 'a+') as f:
            print(wid,'\t',float(pos[wid]),'\n',file=f)


        # take observation wiht one-sided noise:
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
            result = {'x':[bestx], 'y':[y], 'par': [beta_hat,pos], 'pre-eval':True}
        else:
            result = {'x':bestx,'y':y,'par':[beta_hat,pos]}

        with open(qinfo.result_file, 'wb') as f:
            pkl.dump(result, f)


        print('NATS. trial: ',self.trl,' worker: ',wid,'recv_time: ',recv_time,'finished writing')

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
