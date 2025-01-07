"""
@author: Claudio Bellei
-----------------------
This code has been tested on pymc3 version 3.2
"""

import pymc3 as pm
import numpy as np
from scipy.stats import norm
import theano
from theano import tensor as T
from pymc3.math import switch
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict

matplotlib.rc('font', size=10)
matplotlib.rc('font', family='Arial')

np.random.seed(100)

class cpt_bayesian:
    def __init__(self,type="normal-mean"):
        self.type = type
        self.labels = {"xlabel":"Time","ylabel":"Value"}
        self.data = np.array([])

        #define different template problems
        if type=="1-cpt-mean" or type=="normal-mean":
            self.case = 0
            self.ncpt = 1
            size = [2500,2500]
            scale = [50,50] #standard deviation of distribution function
            mu = [1000,1020] #mean of normal for first part
        elif type=="1-cpt-var" or type=="normal-var":
            self.case = 1
            self.ncpt = 1
            size = [2500,2500]
            scale = [10,20] #standard deviation of normal distribution function
            mu = [1000,1000] #mean of normal distributions
            #define random variates
        elif type=="2-cpt-mean":
            self.case = 2
            self.ncpt = 2
            size = [1000,1000,1000]
            scale = [30,30,30] #standard deviation of distribution function
            mu = [1000,1100,800] #mean of normal distributions
            #probability of failing the test
        elif type=="3-cpt-mean":
            self.case = 3
            self.ncpt = 3
            size = [1000,1000,500,1000]
            mu = [1000,1100,800,1020]
            scale = [30,30,30,30]
        else:
            print("invalid choice of type")
            print("options are: normal-mean, normal-var,2-cpts-mean, 3-cpts-mean")

        #stack data into single numpy array
        for i in range(len(size)):
            self.data = np.hstack([self.data,norm.rvs(loc=mu[i],size=size[i],scale=scale[i])])
        self.N = len(self.data)

    def plot_data(self,type="ts",p=None):
        fig = plt.figure(figsize=(10,6))
        n = len(self.data)
        marker = ''
        linestyle = '-'
        plt.plot(np.arange(1,n+1),self.data,ls=linestyle,marker=marker)
        plt.xlabel(self.labels["xlabel"],fontsize=15)
        plt.ylabel(self.labels["ylabel"],fontsize=15)
        plt.ylim([0.9*np.min(self.data),1.1*np.max(self.data)])
        plt.tick_params(axis='both', which='major', labelsize=15)
        fig.set_tight_layout(True)
        if type=="cpt":
            tau = p[0]
            m1 = p[1]
            m2 = p[2]
            plt.plot([0,tau-1],[m1,m1],'r',lw=2)
            plt.plot([tau,n],[m2,m2],'r',lw=2)
            plt.plot([tau,tau],[0.9*np.min(self.data),1.1*np.max(self.data)],'r--',lw=2)
            filename = self.type + "-cpt.png"
            plt.savefig(filename,format="png")
        filename = self.type + ".png"
        plt.savefig(filename,format="png")
        plt.show()

    #find changepoint(s) using pymc3 library
    def find_changepoint(self):
        niter_vec = [5000,2000,2000,3000]
        niter = niter_vec[self.case]
        data = self.data
        #initialize defaultdict for change point priors
        tau = defaultdict(list)
        #initialize defaultdict for uniform priors
        u = defaultdict(list)
        #time array
        t = np.arange(0,self.N)

        with pm.Model() as model: # context management

            #define uniform priors for mean values/standard deviation
            #depending on the type of problem
            for i in range(self.ncpt+1):
                if( not self.type=="1-cpt-var" and not self.type=="normal-var"):
                    varname = "mu" + str(i+1)
                    u[i] = pm.Uniform(varname,650,1200)
                else:
                    varname = "sigma" + str(i+1)
                    u[i] = pm.Uniform(varname,5.,60.)

            #define switch function
            for i in range(self.ncpt):
                varname = "tau" + str(i+1)
                if (i==0):
                    tmin = t.min()
                    switch_function = u[0]
                else:
                    tmin = tau[i-1]
                tau[i] = pm.DiscreteUniform(varname,tmin,t.max())
                switch_function = T.switch(tau[i]>=t,switch_function,u[i+1])

            #we are finally in a position to define the mu and sigma random variables
            if(not self.type=="1-cpt-var" and not self.type=="normal-var"):
                mu = switch_function
                sigma = pm.Uniform("sigma",1,60)
            else:
                mu = pm.Uniform("mu",600,1500)
                sigma = switch_function

            #define log-likelihood function
            logp = - T.log(sigma * T.sqrt(2.0 * np.pi)) \
                  - T.sqr(data - mu) / (2.0 * sigma * sigma)
            def logp_func(data):
                return logp.sum()

            #evaluate log-likelihood given the observed data
            L_obs = pm.DensityDist('L_obs', logp_func, observed=data)

            self.trace = pm.sample(niter, random_seed=123, progressbar=True)
            #define initial condition for algorithm, based on MAP
            #start = pm.find_MAP()
            #start iterations
            #self.trace = pm.sample(niter, start=start, random_seed=123, progressbar=True)

    def plot_results(self):
        if(self.type=="3-cpt-mean"):
            _ = pm.traceplot(self.trace[2000:])
        else:
            _ = pm.traceplot(self.trace[1400:])
        filename = self.type + "-results.png"
        plt.savefig(filename,format="png")
        plt.show()

    def plot_theory(self):
        if(self.type=="1-cpt-mean" or self.type=="normal-mean"):
            N = len(self.data)
            x = self.data
            #equation (11) in blog post
            logP = -0.5*(N-2)*np.log( np.asarray([(tau*(N-tau))**(-0.5)* \
                ( np.sum(x**2) - np.sum(x[1:tau])**2/tau - np.sum(x[tau:N])**2/(N-tau) ) \
                for tau in np.arange(1,N)]) )
            #sum a "random" number to avoid underflow
            logP += - np.max(logP) - 1
            #now get posterior probability
            P = np.exp(logP)
            #and finally normalize to one
            P /= np.trapz(P)

            #now plot result
            #1. theory results
            plt.plot(np.arange(1,N),P,'r-',linewidth=3,label="eq. (11)")
            #2. pymc3 results (note that I am adding +1 to the trace)
            results, edges = np.histogram(self.trace["tau1"]+1,bins=50,normed=True)
            binWidth = edges[1] - edges[0]
            plt.bar(edges[:-1], results, binWidth,label="pymc3")
            plt.xlim([2400,2550])
            plt.tick_params(axis='both', which='major', labelsize=20)
            plt.xlabel("Time",fontsize=20)
            plt.ylabel(r"$\mathbb{P}(\tau|d)$",fontsize=20)
            plt.legend(prop={'size':15})
            plt.savefig("pymc3_vs_theory.png",format="png")
            plt.show()
        else:
            print("Can only plot theory result for single change point in mean")
            print("--Aborting--")