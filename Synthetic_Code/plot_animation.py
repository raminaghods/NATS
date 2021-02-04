"""
Code for submitted work:

``Multi Agent Active Search using Realistic Depth-Aware Noise Model'',
submitted to 2020 Conference on Robot Learning
author: anonymous

please do not distribute. The code will become public upon acceptance of the paper.

"""

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import animation, ticker
import numpy as np
import seaborn as sns; sns.set()
import os


def plot_animation(n1,n2,k,T,mu,dir):

    n=n1*n2
    beta_hat_ar = []
    wid_beta_hat_ar = []
    sigma_ar = []
    wid_sigma_ar = []
    with open(os.path.join(dir,'posteriorfile.txt'),'r') as f:
        line0 = f.readline()
        array_str = line0.strip().split(' ')
        true_beta = np.array(array_str).astype('int')
        data = f.read()
        splat = data.strip().split('\n\n')
        for number, paragraph in enumerate(splat, 1):
            aa = paragraph.strip().split('\t')
            # print(aa)
            wid,l = paragraph.strip().split('\t')
            array_str = l.strip().split(' ')
            if number % 2 == 1:
                beta_hat_ar.append(np.array(array_str).astype('float32'))
                wid_beta_hat_ar.append(float(wid))
            else:
                sigma_ar.append(np.array(array_str).astype('float32'))
                wid_sigma_ar.append(float(wid))
    beta_tilde_ar = []
    wid_beta_tilde_ar = []
    sensors_ar = []
    wid_sensor_ar = []
    with open(os.path.join(dir,'resfile.txt'),'r') as f:
        data = f.read()
        splat = data.strip().split('\n\n')
        for number, paragraph in enumerate(splat, 1):
            aa = paragraph.strip().split('\t')
            # print(aa)
            wid,l = paragraph.strip().split('\t')
            array_str = l.strip().split(' ')
            if number % 2 == 1:
                beta_tilde_ar.append(np.array(array_str).astype('float32'))
                wid_beta_tilde_ar.append(float(wid))
            else:
                sensors_ar.append(np.array(array_str).astype('int'))
                wid_sensor_ar.append(float(wid))

    pos0_ar = []
    pos1_ar = []
    with open(os.path.join(dir,'positionfile.txt'),'r') as f:
        data = f.read()
        splat = data.strip().split('\n\n')
        for number, paragraph in enumerate(splat, 1):
            aa = paragraph.strip().split('\t')
            # print(aa)
            wid,l = paragraph.strip().split('\t')
            array_str = l.strip().split(' ')
            if(float(wid)==0.0):
                pos0_ar.append(np.array(array_str).astype('float32'))
            else:
                pos1_ar.append(np.array(array_str).astype('float32'))



    plt.close('all')
    fig, ((ax0,ax1),(ax2,ax3)) = plt.subplots(2, 2)


    #fig = plt.figure()
    #ax0 = fig.add_subplot(121, aspect='equal', autoscale_on=False,xlim=(0,n1),ylim=(0,n2))
    #ax1 = fig.add_subplot(122, aspect='equal', autoscale_on=False,xlim=(0,n1),ylim=(0,n2))


    # Major ticks every 20, minor ticks every 5
    major_ticksx = np.arange(-0.5, n1, 4)
    minor_ticksx = np.arange(-0.5, n1, 1)
    major_ticksy = np.arange(-0.5, n2, 4)
    minor_ticksy = np.arange(-0.5, n2, 1)

    ax0.set_xticks(major_ticksx)
    ax0.set_xticks(minor_ticksx, minor=True)
    ax0.set_yticks(major_ticksy)
    ax0.set_yticks(minor_ticksy, minor=True)

    # And a corresponding grid
    ax0.grid(which='both',color='grey')

    # Or if you want different settings for the grids:
    ax0.grid(which='minor', alpha=0.5)
    ax0.grid(which='major', alpha=1)

    ax1.set_xticks(major_ticksx)
    ax1.set_xticks(minor_ticksx, minor=True)
    ax1.set_yticks(major_ticksy)
    ax1.set_yticks(minor_ticksy, minor=True)


    # And a corresponding grid
    ax1.grid(which='both',color='grey')

    # Or if you want different settings for the grids:
    ax1.grid(which='minor', alpha=0.5)
    ax1.grid(which='major', alpha=0.75)

    for pos in ['top', 'bottom', 'right', 'left']:
        ax0.spines[pos].set_edgecolor('grey')
        ax1.spines[pos].set_edgecolor('grey')

    ax0.set_title('Agent 0')# \n  k=%d, T=%d'%(k,T))
    ax1.set_title('Agent 1')# \n k=%d, T=%d'%(k,T))
    ax0.set_ylabel('posterior samples \n')

    ax2.set_xticks(major_ticksx)
    ax2.set_xticks(minor_ticksx, minor=True)
    ax2.set_yticks(major_ticksy)
    ax2.set_yticks(minor_ticksy, minor=True)

    # And a corresponding grid
    ax2.grid(which='both',color='grey')

    # Or if you want different settings for the grids:
    ax2.grid(which='minor', alpha=0.5)
    ax2.grid(which='major', alpha=1)

    ax3.set_xticks(major_ticksx)
    ax3.set_xticks(minor_ticksx, minor=True)
    ax3.set_yticks(major_ticksy)
    ax3.set_yticks(minor_ticksy, minor=True)


    # And a corresponding grid
    ax3.grid(which='both',color='grey')

    # Or if you want different settings for the grids:
    ax3.grid(which='minor', alpha=0.5)
    ax3.grid(which='major', alpha=0.75)

    for pos in ['top', 'bottom', 'right', 'left']:
        ax2.spines[pos].set_edgecolor('grey')
        ax3.spines[pos].set_edgecolor('grey')

    ax2.set_xlabel('posterior mean')
    ax3.set_xlabel('posterior variance')

    ax0.set_xticklabels(['']*n1)
    ax0.set_yticklabels(['']*n2)
    ax1.set_xticklabels(['']*n1)
    ax1.set_yticklabels(['']*n2)
    ax2.set_xticklabels(['']*n1)
    ax2.set_yticklabels(['']*n2)
    ax3.set_xticklabels(['']*n1)
    ax3.set_yticklabels(['']*n2)


    im0 = ax0.imshow(np.reshape(beta_tilde_ar[0],(n2,n1)),cmap="RdPu",vmin=0,vmax=mu)
    im1 = ax1.imshow(np.reshape(beta_tilde_ar[0],(n2,n1)),cmap="RdPu",vmin=0,vmax=mu)
    im2 = ax2.imshow(np.reshape(beta_hat_ar[0],(n2,n1)),cmap="RdPu",vmin=0,vmax=mu)
    im3 = ax3.imshow(np.reshape(sigma_ar[0],(n2,n1)),cmap="YlGn",vmin=0,vmax=0.06)#,norm=matplotlib.colors.LogNorm())

    #im3 = ax0.imshow(np.reshape(beta_tilde_ar[0],(n1,n2)),cmap="Reds",alpha=0.1)
    #im3 = ax0.fill(np.zeros((n1,n2),dtype=bool), 1, hatches=['//'], alpha=0)
    #scat0 = ax0.scatter([],[],marker='o',linewidth=12,color='red')

    scat0 = ax0.scatter(np.array([]),np.array([]),marker='o',linewidth=1,color='cyan')
    scat1 = ax1.scatter(np.array([]),np.array([]),marker='o',linewidth=1,color='cyan')

    true_pos = np.zeros((n,1))
    true_pos[true_beta] = mu
    (ytruepos,xtruepos) = np.nonzero(np.reshape(true_pos,(n2,n1)))
    ax0.scatter(xtruepos, ytruepos, marker='x', color='black',linewidth=1)
    ax1.scatter(xtruepos, ytruepos, marker='x', color='black',linewidth=1)
    ax2.scatter(xtruepos, ytruepos, marker='x', color='black',linewidth=1)
    ax3.scatter(xtruepos, ytruepos, marker='x', color='black',linewidth=1)


    line0, = ax0.plot(np.array([]),np.array([]),color='blue',linewidth=0.5)
    line1, = ax1.plot(np.array([]),np.array([]),color='blue',linewidth=0.5)
    fig.colorbar(im0,ax=[ax0])
    fig.colorbar(im1,ax=[ax1])
    fig.colorbar(im2,ax=[ax2])
    fig.colorbar(im3,ax=[ax3])



    ##def init():
    ##
    #
    ##
    ##    return (fig, )
    #

    def animate(i):
        if(wid_beta_tilde_ar[i]==0.0 and wid_sensor_ar[i]==0.0):
            im0.set_array(np.reshape(beta_tilde_ar[i],(n2,n1)))
            sensor = np.zeros((n,1))
            sensor[sensors_ar[i]] = mu
            (y,x) = np.nonzero(np.reshape(sensor,(n2,n1)))
            aa = np.concatenate((np.reshape(x,(-1,1)),np.reshape(y,[-1,1])),axis=1)
            scat0.set_offsets(aa)
            im2.set_array(np.reshape(beta_hat_ar[i],(n2,n1)))
            im3.set_array(np.reshape(sigma_ar[i],(n2,n1)))
            wids = np.array(wid_sensor_ar)[0:i+1]
            count = wids[wids==0.0].shape[0]
            l0 = np.array(pos0_ar)[0:count] % n1
            line0.set_data(l0,(np.array(pos0_ar)[0:count]-l0)/n1)
    #        map1 = ax1.plot([], [])
        elif(wid_beta_tilde_ar[i]==1.0 and wid_sensor_ar[i]==1.0):
            im1.set_array(np.reshape(beta_tilde_ar[i],(n2,n1)))
            sensor = np.zeros((n,1))
            sensor[sensors_ar[i]] = mu
            (y,x) = np.nonzero(np.reshape(sensor,(n2,n1)))
            scat1.set_offsets(np.concatenate((np.reshape(x,(-1,1)),np.reshape(y,[-1,1])),axis=1))
            im2.set_array(np.reshape(beta_hat_ar[i],(n2,n1)))
            im3.set_array(np.reshape(sigma_ar[i],(n2,n1)))
            wids = np.array(wid_sensor_ar)[0:i+1]
            count = wids[wids==1.0].shape[0]
            l1 = np.array(pos1_ar)[0:count] % n1
            line1.set_data(l1,(np.array(pos1_ar)[0:count]-l1)/n1)
    #        map0 = ax0.plot([], [])
        return [im0,im1]

    ani = animation.FuncAnimation(fig, animate, frames=T,interval=5000, blit=True)#, repeat=True, repeat_delay=2000)
    ani.save(os.path.join(dir,'AlgDetails_animation_%dx%d_k_%2.0d_T_%d.mp4'%(n1,n2,k,T)), fps=1, extra_args=['-vcodec', 'libx264'])

    print('animation saved in your directory!')
if __name__ == "__main__":

    plot_animation2(16,16,6,150,1.0,"Files")
