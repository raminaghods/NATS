'''
Code for submitted work:

``Multi Agent Active Search using Realistic Depth-Aware Noise Model'',
submitted to 2020 Conference on Robot Learning
author: anonymous

please do not distribute. The code will become public upon acceptance of the paper.

In this file, we are coding the RSI algorithm from reference:
Ma, Y., Garnett, R., and Schneider, J. Active search for
sparse signals with region sensing. In Thirty-First AAAI
Conference on Artificial Intelligence, 2017.

We use this code to compare RSI algorithm from the aforementioned reference to our proposed SPATS and LATSI algorithms.

'''

import numpy as np
import math
import pickle as pkl
import scipy.stats as ss

import copy
import os

class RSI(object):

    def __init__(self, beta,n1, mu, theta2, noise_vec, lmbd, EMitr, err, n_agents,trl):
        self.mu = mu
        self.beta = beta
        self.k = np.count_nonzero(self.beta)
        self.theta2 = theta2
        self.noise_vec = noise_vec
        self.lmbd = lmbd
        self.EMitr = EMitr
        self.trl = trl
        self.n_agents = n_agents
        self.n = beta.shape[0]
        self.n1 = n1
        self.n2 = int(self.n/self.n1)
        self.L = int(self.n/2)
        self.M = int(self.n / self.L)

        self.rng = np.random.RandomState(trl)

        self.gamma = self.rng.rand(self.M)
        self.B = self.rng.rand(self.L,self.L)
        self.Sig0 = np.kron(np.diag(self.gamma),self.B)
        self.err = err
#
#        print('init RSI')

    def set_beta(self,beta):
        self.beta = beta

    def sample_from_prior(self, num_samples):

        beta_tilde = np.zeros((num_samples, self.n, 1))

        for idx in range(num_samples):

            beta_tilde[idx] = self.rng.laplace(scale=1/self.lmbd,size=(self.n,1))

        return beta_tilde



    def getPosterior(self, X, Y, beta_rsi, recv_time):#rng, n, X, Y, EMitr=1):

        ######### posterior computation RSI #########
        pi_0 = np.ones((self.n,1),dtype=np.float64) * 1. / self.n
        beta_hat_rsi = np.zeros((self.n,1))
#        print('X shape: ',X.shape)

        beta_rsi = copy.deepcopy(self.beta)

        for i in range(X.shape[0]):
            for j in range(self.n):
                b = np.zeros((self.n,1))
                b[j,:] = self.mu
                # print('X[i] shape: ', X[i].shape)
                #assert X[i].shape == (1,32)
                # print('shape: ', Y[i].shape)
                pi_0[j] = np.float64(pi_0[j] * ss.norm(0,np.sqrt(Y[i,1])).pdf(Y[i,0] - np.dot(X[i], b)))
            pi_0 /= np.sum(pi_0)

        # return pi_0, pi_0, beta_rsi, None, None
            if np.amax(pi_0) == 0.:
                print('1-process ',os.getpid(),'with recv_time',recv_time,' would raise ValueError')
                break

                # raise ValueError('pi_0 max value 0.!')
            maxidxs = np.argwhere(pi_0 == np.amax(pi_0))
            eps = 0.
            for ids in maxidxs:
                eps += 1 - pi_0[ids][0]
            if(eps<self.err):
                # idxs = []
                for m in maxidxs:
                    beta_hat_rsi[m[0],:] = self.mu

                    beta_rsi[m[0],:] = 0.

                    # idxs.append(m[0])

                    # pi_0[m[0]] = 0.
                detected = np.count_nonzero(beta_hat_rsi)
                if(detected >= self.k):
                    break
#                print("so far detected: ",detected," left:",np.count_nonzero(beta_rsi))
                if np.count_nonzero(beta_rsi) == 0:
                    break
                #pi_0 /= np.sum(pi_0)

                idxs = [ii for ii in range(self.n) if ii not in np.nonzero(beta_hat_rsi)[0]]
                for idx in idxs:
                    b = copy.deepcopy(beta_hat_rsi)
                    b[idx,:] = self.mu
                    pi_0[idx] = 1./self.n
                    for t in range(i+1):
                        pi_0[idx,:] = np.float64(pi_0[idx,:] * ss.norm(0,np.sqrt(Y[t,1])).pdf(Y[t,0] - np.dot(X[t],b)))
                        # if(pi_0[idx,:]>1e5):
                        #     pi_0/= 1e3

                        # print(ss.norm(0,np.sqrt(Y[t,1])).pdf(Y[t,0] - np.dot(X[t],b)))
                #pi_0 = np.full((self.n,1), 1./(self.n - detected))
                # pi_0[np.nonzero(beta_hat_rsi)[0],:] = 0.
                # pi_0[idxs] = 0.
                if (np.amax(pi_0) == 0):
                    print('2-process ',os.getpid(),' would raise ValueError')
                    print(recv_time)
                    break
                    # raise ValueError('pi_0 max value 0.!')
#                print('sum:',np.sum(pi_0))

                if(math.isinf(np.amax(pi_0)) or math.isnan(np.sum(pi_0))):
                    pi_0 = np.nan_to_num(pi_0)
                    print('inf or nan value for pi_0')

                pi_0 /= np.nansum(pi_0)

                #print('sum: ',np.sum(pi_0))

                #beta_rsi = beta_rsi - beta_hat_rsi

        print('returning beta_hat')
        _ = 1
        return _, pi_0, beta_rsi,beta_hat_rsi,_

    def ActiveSearch(self,points_dict,qinfo):#beta,mu,theta2,sigma2,lmbd,EMitr,T,trl=np.random.randint(1000),X,Y):

        recv_time = qinfo.send_time
        self.rng = np.random.RandomState(self.trl+recv_time)
        wid =  qinfo.worker_id
        if qinfo.compute_posterior:
            X = points_dict['X']
            Y = points_dict['Y']
            [_,pos] = points_dict['par']
            _, pi_0,beta_rsi,beta_hat_rsi,_ = self.getPosterior(X,Y,_,qinfo.send_time)
            # print('RSI\\1. trial: ',self.trl,' worker # ',wid,' recv_time ',recv_time,' finished posterior compute')
        else:#this branch True on initial evaluations for agents with samples from prior
            beta_rsi = self.beta
            pi_0 = np.ones((self.n,1)) * 1. / self.n
            pos = np.zeros((self.n_agents))
            beta_hat_rsi = np.zeros((self.n,1))

        k = np.count_nonzero(self.beta)


        max_reward = -math.inf
        bestx = np.zeros((self.n,1))

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

                    # RSI information Gain:
                    p_0 = np.sum(pi_0[[ll for ll in range(self.n) if ll not in nonzero_idx]])
                    p_1 = np.sum(pi_0[nonzero_idx])
                    lamda = self.mu * 1.
                    IG = 0
                    if(np.sum(pi_0) != 0):
                        IG += -(p_0 * np.log(p_0 * ss.norm(0,1).pdf(0) + p_1 * ss.norm(0,1).pdf(-lamda)))
                        IG += -(p_1 * np.log(p_0 * ss.norm(0,1).pdf(lamda) + p_1 * ss.norm(0,1).pdf(0)))

                    loss[h,l,d] = -IG
                    if (qinfo.compute_posterior):
                        dist[h,l,d] = (np.absolute(oldl-l)**2+np.absolute(oldh-h)**2)
                    else:
                        dist[h,l,d] = 0.001

                    avg_loss += loss[h,l,d]
                    avg_dist += dist[h,l,d]
                    count += 1

        avg_loss /= count
        avg_dist /= count

        reward = - loss#/avg_loss - dist/avg_dist
        [hm,lm,dm] = np.unravel_index(reward.argmax(), reward.shape)
        bestx,bestidx,best_noise_var = self.create_directional_sensor(lm,hm,dm)
        pos[wid] = float(hm*self.n1+lm)

        # beta_tilde = beta_hat_rsi # temporary fake value for beta_tilde
        # with open('resfile.txt', 'a+') as f:
        #     print(wid,'\t',' '.join(map(str, np.squeeze(np.round(beta_tilde,2)))),'\n',file=f)
        #     print(wid,'\t',' '.join(map(str,bestidx)),'\n',file=f)
        #
        # with open('posteriorfile.txt', 'a+') as f:
        #     print(wid,'\t',' '.join(map(str, np.squeeze(np.round(beta_hat_rsi,2)))),'\n',file=f)
        #     print(wid,'\t',' '.join(map(str, np.squeeze(np.round(pi_0,2)))),'\n',file=f)
        #
        # with open('positionfile.txt', 'a+') as f:
        #     print(wid,'\t',float(pos[wid]),'\n',file=f)

        #%% take a new observation
        # one-sided noise:
        epsilon = np.abs(np.multiply(self.rng.randn(bestx.shape[0],1),np.reshape(np.sqrt(best_noise_var),(-1,1))))
        epsilon_neg = -1*np.abs(np.multiply(self.rng.randn(bestx.shape[0],1),np.reshape(np.sqrt(best_noise_var),(-1,1))))
        obs = np.matmul(bestx,self.beta)
        epsilon[obs>0] = epsilon_neg[obs>0]
        y_temp = np.squeeze(np.matmul(bestx,beta_rsi)+epsilon)
        y_temp[y_temp<0.0] = 0.0
        y_temp[y_temp>self.mu] = self.mu
        y = np.zeros((bestx.shape[0],2))
        y[:,0] = y_temp
        y[:,1] = best_noise_var

        if qinfo.compute_posterior:
            X = np.append(X,bestx,axis=0)
            Y = np.append(Y,y,axis=0)
        else:
            X = bestx
            Y = y

        _,_,_,beta_hat_rsi,_ = self.getPosterior(X,Y,None,recv_time)

        if not qinfo.compute_posterior:
            #X = [np.transpose(X), bestx]#np.append(X, np.transpose(bestx),axis=0)
            #print('len X: ',len(X))
            #Y = [Y, y]#np.append(Y, y, axis=0)
            result = {'x':[bestx], 'y':[y], 'par':[beta_hat_rsi,pos], 'pre-eval':True}
        else:
            result = {'x':bestx,'y':y,'par':[beta_hat_rsi,pos]}

        # print('RSI\\3. trial: ',self.trl,' worker: ',wid,'recv_time: ',recv_time,'finished compute')
        with open(qinfo.result_file, 'wb') as f:
            pkl.dump(result, f)


        print('RSI. trial: ',self.trl,' worker: ',wid,'recv_time: ',recv_time,'finished writing')
        # print('pi_0: ',pi_0)
        # print('beta_hat_rsi: ',beta_hat_rsi)
        # print('agent ', wid, ' returned: ',result)

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
