import pickle
import itertools
import math
import numpy as np
import time
from numpy import savetxt
import matplotlib.pyplot as plt

class Observable:
    def __init__(self, L,Nboot,binsize):
        self.L = L # Lattice Size = (L,L)
        self.Nboot = Nboot # No. of times Bootstrapping is done.
        self.binsize = binsize # say 100 
        
    def mean_of_absolute_phi(self,conf):
        sample_mean = []
        data = conf.reshape(-1,self.L*self.L).mean(axis = 1)
        abs_data = np.absolute(data).reshape(-1,self.binsize)
        
        for i in range(self.Nboot):
            x=abs_data[np.random.randint(0,abs_data.shape[0],abs_data.shape[0])]
            sample_mean.append(x.mean(axis=(0,1)))
            
        return np.mean(sample_mean), np.std(sample_mean,ddof=1)
    
    def mean_of_sqr_phi(self,conf):
        sample_mean = []
        data = conf.reshape(-1,self.L*self.L).mean(axis = 1)
        abs_data = (np.absolute(data)**2).reshape(-1,self.binsize)
        
        for i in range(self.Nboot):
            x=abs_data[np.random.randint(0,abs_data.shape[0],abs_data.shape[0])]
            sample_mean.append(x.mean(axis=(0,1)))
            
        return np.mean(sample_mean), np.std(sample_mean,ddof=1)
    
   
    def twopoint_susceptibility(self,conf):
        
        sample_mean = []        
        cfgs=conf.reshape(-1,self.L,self.L)
             
        C=0
        for x in range(self.L):
            for y in range(self.L):
                C = C + cfgs*np.roll(cfgs, (-x, -y), axis=(1,2))
        X = np.mean(C, axis=(1,2))
        
        config = X.reshape(-1,self.binsize)
        for i in range(self.Nboot):
            idx = np.random.randint(0,config.shape[0],config.shape[0])
            x1=config[idx]
            sample_mean.append(x1.mean(axis=(0,1)))
        return np.mean(sample_mean), np.std(sample_mean,ddof=1)
    
    def autocorrelation_analysis(self,conf,window):
        # initial processing
        cfgs = conf.reshape(-1,self.L*self.L)
        data = (cfgs.mean(axis=1))**2
        data_size = len(data)
        avg = np.average(data)
        data_centered = data - avg

        # auto-covariance function
        autocov = np.empty(window)
        N_tau   = np.empty(window)
        for j in range(window):
            autocov[j] = np.dot(data_centered[:(data_size - j)], data_centered[j:])
            N_tau[j] = data_size - j
        autocov /= N_tau

        # autocorrelation function
        acf = autocov / autocov[0]

        # integrate autocorrelation function
        j_max_v = np.arange(window)
        tau_int_v = np.zeros(window)
        for j_max in j_max_v:
            tau_int_v[j_max] = 0.5 + np.sum(acf[1:j_max + 1])

        
        tau_int = tau_int_v[j_max]
#         N_eff = data_size / (2 * tau_int)
#         sem = np.sqrt(autocov[0] / N_eff)

#         fig, (ax1, ax2) = plt.subplots(2,figsize=(10,10))
#         fig.suptitle('Autocorrelation = %f ' %tau_int_v[-1:])
#         ax1.plot(j_max_v, acf)
#         ax2.plot(j_max_v, tau_int_v)
#         ax1.set_xlabel( r'$\tau $ :Distance between two mc configuration')
#         ax1.set_ylabel('Autocorrelation Function')
#         ax2.set_xlabel('T_limit :maximum' r'$\tau $ ')
#         ax2.set_ylabel('Integrated autocorrelation')
#         plt.show()


#         # print out stuff
#         print(f"Mean value: {avg:.4f}")
#         print(f"Standard error of the mean: {sem:.4f}")
#         print(f"Integrated autocorrelation time: {tau_int:.3f} time steps")
#         print(f"Effective number of samples: {N_eff:.1f}")

        return tau_int


    
    def Correlation_function(self,x):
        sample_mn = []
        bin_mn = []
        x=x.reshape(-1,self.binsize,self.L*self.L)
        for x1 in x:
            bin_mn.append(self.corr_results(x1))
        bin_mn=np.array(bin_mn)
        print(bin_mn.shape)
        for i in range(self.Nboot):
          idx=np.random.randint(0,bin_mn.shape[0],bin_mn.shape[0])
          x1=bin_mn[idx]
          sample_mn.append(x1.mean(axis=0))
        sample_mn=np.array(sample_mn)
        print(sample_mn.shape)

        mn=np.mean(sample_mn,axis=0)
        sd=np.std(sample_mn,axis=0,ddof=1)
        #CI=sample_mn[ci1]-sample_mn[ci2]
        return mn , sd

       


    def compute_twopts(self,cfgs, xspace=1, tspace=1):
        start = time.time()
        corrs = []
        Nd = len(cfgs.shape[1:])
        spatial_axes = tuple(range(0,Nd-1))
        # smear = make_smear_corr(unwrap_smear_x, unwrap_smear_t)
        for i,c in enumerate(cfgs):
            #if verbose and i % 10 == 0:
                #print('Cfg {} / {} ({:.2f}s)'.format(
                   # i+1, cfgs.shape[0], time.time()-start))
            corr = np.array(self.all_corrs(c, xspace, tspace))
            corr = np.mean(corr, axis=0)
            corrs.append(corr)
        #print('Two-points done')
        return corrs
    def all_corrs(self,phi, xspace, tspace):
        corrs = []
        coord_ranges = []
        for Lx in phi.shape[:-1]:
            assert(Lx % xspace == 0)
            coord_ranges.append(range(0, Lx, xspace))
        assert(phi.shape[-1] % tspace == 0)
        coord_ranges.append(range(0, phi.shape[-1], tspace))
        all_axes = tuple(range(len(phi.shape)))
        for src in itertools.product(*coord_ranges):
            shift = tuple(-np.array(src))
            corr = np.conj(phi[src]) * np.roll(phi, shift, axis=all_axes)
            corrs.append(corr)
        return np.array(corrs)
    def corr_results(self,dta):  
        phi1=dta.reshape(-1,self.L,self.L) 
        xspace=1
        tspace=1
        corrs = self.compute_twopts(phi1, xspace, tspace)
        twopt1 = np.array(corrs)
        twopt_mean=np.mean(twopt1, axis=0)
        twopt_1=np.mean(twopt_mean, axis=0)
        dt=phi1.mean(axis=0)
        C=0
        for x in range(self.L):
            for y in range(self.L):
                C = C + dt*np.roll(dt, (-x, -y), axis=(0,1))
        #print(C)
        twopt_2=np.mean(C/(self.L*self.L),axis=0)
        crr=twopt_1-twopt_2
        return crr
    def Autocorrelation(self,Data_full,Final_Tmax,Gap,figure):   
        Data=Data_full.reshape(-1,self.L*self.L)
        m=np.mean(Data, axis=-1)
        m=m**2
        sig=m
        N=len(m)    
        limit=np.arange(1,Final_Tmax,Gap)
        A_int1=np.zeros(len(limit))
        ind=0
        for i in limit:
            tau=np.arange(0,i,1)
            A=np.zeros(i)
            for ta in tau:
                x1=0.0
                x2=0.0
                x=0.0
                for j in range(N-ta):
                    x2=x2+sig[j]
                    x1=x1+sig[j+ta]
                    x=x+sig[j]*sig[j+ta]
                x3= (1.0/(N-ta))*x2*x1
                x4=x
                A[ta]=(1.0/(N-ta))*(x4-x3)
            A_int=0        
            for ta in tau:  
                 A_int=A_int+A[ta]
            A_int1[ind]=.5+A_int/A[0]
            ind=ind+1;
        print(A_int1[-1:]) 
        if figure==True:   
            fig, (ax1, ax2) = plt.subplots(2,figsize=(10,10))
            fig.suptitle('Autocorrelation = %f ' %A_int1[-1:])
            ax1.plot(tau, A)
            ax2.plot(limit, A_int1)
            ax1.set_xlabel( r'$\tau $ :Distance between two mc configuration')
            ax1.set_ylabel('Autocorrelation Function')
            ax2.set_xlabel('T_limit :maximum' r'$\tau $ ')
            ax2.set_ylabel('Integrated autocorrelation')
            plt.show()
        return A_int1[-1]