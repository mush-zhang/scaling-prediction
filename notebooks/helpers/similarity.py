#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

from collections import Counter

from sklearn import preprocessing
import functools

from tslearn.metrics import dtw
from tslearn.metrics import lcss

from scipy.spatial.distance import pdist

import ruptures as rpt
import sdt
from sdt.helper import numba
from sdt import changepoint
from sdt.changepoint import bayes_online as online

from statsmodels.tsa.stattools import adfuller

from helpers.scale_data import ScaleData


# import nbimporter
# import expr_data


# In[2]:


def update_class(main_class=None, exclude=("__module__", "__name__", "__dict__", "__weakref__")):
    """Class decorator. Adds all methods and members from the wrapped class to main_class

    Args:
    - main_class: class to which to append members. Defaults to the class with the same name as the wrapped class
    - exclude: black-list of members which should not be copied
    """

    def decorates(main_class, exclude, appended_class):
        if main_class is None:
            main_class = globals()[appended_class.__name__]
        for k, v in appended_class.__dict__.items():
            if k not in exclude:
                setattr(main_class, k, v)
        return main_class

    return functools.partial(decorates, main_class, exclude)


# ## Similarity Computation
# 
# For each type of data (plan, event), we try to calculate similarity based on:
# 
# 1. bined distribution of each feature;
# 2. shape and absolute values of timeseries data

# In[3]:


class Similarity():
    def __init__(self, data, plan_mtxs_splitted, plan_col_ranges, perf_mtxs_splitted, perf_col_ranges, num_bins=10):
        self.plan_mtxs_splitted = plan_mtxs_splitted
        self.plan_col_ranges = plan_col_ranges
        self.perf_mtxs_splitted = perf_mtxs_splitted
        self.perf_col_ranges = perf_col_ranges
        self.col_ranges = plan_col_ranges+perf_col_ranges
        
        self.num_bins = num_bins
        self.plan_bined = None
        self.perf_bined = None
        
        self.bined = None
        self.cumulative_bined = None
                
        self.simi_col_mtx = None
        
        self.simi_mtx = None

        self.data = data


# In[4]:


@update_class()
class Similarity():
    '''
    Not used
    '''
    def plot_col_distribution(plan_mtxs_splitted, idx, num_bins=10):
        bined = []
        # for each workload
        for mtx in plan_mtxs_splitted:
            curr_list = mtx[ : , idx]
            curr_col, edges = np.histogram(curr_list, bins=num_bins, range=plan_col_ranges[i], density=True)
            bined.append(hist)

        curr_list = mtx[ : , i]
        curr_col, edges = np.histogram(curr_list, bins=num_bins, range=plan_col_ranges[i], density=True)
        fig, ax = plt.subplots()
        ax.bar(edges[:-1], curr_col, width=np.diff(edges), edgecolor="black", align="edge")
        plt.show()


# In[5]:


@update_class()
class Similarity():
    '''
    Calculate frequency based bins (each bin value range from 0 to 10 and sums to 10)
    '''
    def calc_bined_mtx(self, plan_only=False, perf_only=False):
        self.plan_bined = []
        if not perf_only:
            # for each workload
            for mtx in self.plan_mtxs_splitted:
                if mtx is None:
                    self.plan_bined.append(None)
                    continue
                # for each feature
                curr_bined = []
                for i in range(mtx.shape[1]):
                    curr_list = mtx[ : , i]
                    curr_col, edges = np.histogram(curr_list, bins=self.num_bins, range=self.plan_col_ranges[i], density=True)
                    curr_bined.append(curr_col)
                # hist_mtx: #bins * #features
                hist = np.array(curr_bined, dtype=object).T
                self.plan_bined.append(hist)
        else:
            self.plan_bined = None
            
        if not plan_only:
            self.perf_bined = []
            # for each workload
            for mtx in self.perf_mtxs_splitted:
                if mtx is None:
                    self.perf_bined.append(None)
                    continue
                # for each feature
                curr_bined = []
                for i in range(mtx.shape[1]):
                    curr_list = mtx[ : , i]
                    curr_col, edges = np.histogram(curr_list, bins=self.num_bins, range=self.perf_col_ranges[i], density=True)
                    curr_bined.append(curr_col)
                # hist_mtx: #bins * #features
                hist = np.array(curr_bined, dtype=object).T
                self.perf_bined.append(hist)
        else:
            self.perf_bined = None
        
        if not perf_only and not plan_only:
            self.bined = self.__concate_mtxs()
            # self.col_ranges = np.append(self.plan_col_ranges, self.perf_col_ranges)
        elif perf_only:
            self.bined = self.perf_bined
        elif plan_only:
            self.bined = self.plan_bined
        self.cumulative_bined = [df.cumsum(axis=0) for df in self.bined]


# In[6]:


@update_class()
class Similarity():
    '''
    for 2 matrices, compute l1,1-norm of their differences
    '''
    def __comp_l11_norm(self, A, B):
        # return np.absolute(A - B).sum()
        return np.linalg.norm(np.absolute(A - B), ord='nuc')
    
    '''
    for 2 matrices, compute l2,1-norm of their differences
    '''
    def __comp_l21_norm(self, A, B):
        temp = np.linalg.norm(A - B, ord=2, axis=0)
        assert(len(temp) == A.shape[1])
        return temp.sum()
    
    '''
    for 2 matrices, compute Frobenius-norm of their differences
    '''
    def __comp_fro_norm(self, A, B):
        temp = np.linalg.norm(A - B, ord='fro')
        return temp.sum()
    
    '''
    for 2 matrices, compute sum of Canberra-norm of their differences
    '''
    def __comp_canb_norm(self, A, B):
        temp = []
        for a, b in zip(A.T, B.T):
            curr = pdist(np.stack((a, b)), 'canberra') 
            assert(len(curr) == 1)
            temp.append(curr[0])
        return np.sum(temp)
    
        '''
    for 2 matrices, compute chi2-norm of their differences
    '''
    def __comp_chi2_norm(self, A, B):
        temp = [ np.square(a-b)/(a+b) for a, b in zip(A.T, B.T)]
        return np.sum(temp)

    '''
    for 2 matrices, compute correlation-norm of their differences
    '''
    def __comp_corr_norm(self, A, B):
        temp = []
        for a, b in zip(A.T, B.T):
            curr = pdist(np.stack((a, b)), 'correlation') 
            assert(len(curr) == 1)
            temp.append(curr[0])
        return np.sum(temp)
    
    '''
    for 2 matrices, compare their column-wise distributions
    '''
    def __comp_col_distribution_fp(self, A, B):
        return np.sum(np.absolute(A - B), axis=0)
    
    '''
    for 2 matrices, compute their dtw distance
    '''
    def __comp_dtw(self, A, B):
        return dtw(A, B)
    
    '''
    for 2 matrices, compute their column wise dtw distance
    '''
    def __comp_col_dtw(self, A, B):
        total = 0
        for a, b in zip(A.T, B.T):
            total += dtw(a, b)
        return total
    
    '''
    for 2 matrices, compute their lcss score
    according to doc: the values returned by LCSS range from 0 to 1, the highest value taken when two time series fully match, and vice-versa.
    '''
    def __comp_lcss(self, A, B):
        return 1- lcss(A, B, eps=0.4)
    
    '''
    for 2 matrices, compute their independent (colwise) lcss score
    according to doc: the values returned by LCSS range from 0 to 1, the highest value taken when two time series fully match, and vice-versa.
    '''
    def __comp_col_lcss(self, A, B):
        total = 0
        for a, b in zip(A.T, B.T):
            total += 1- lcss(a, b, eps=0.1)
        return total
    
    '''
    filter by features
    '''
#     def filter_by_features(self, keep_features, mode='all'):
#         if mode == 'all':
#             all_features = self.data.feature_cols
#         elif mode == 'perf':
#             all_features = self.data.perf_feature_cols
#         elif mode == 'plan':
#             all_features = self.data.plan_feature_cols
            
#         keep_set = set(keep_features)
#         keep_idxs = []
#         for idx in range(len(all_features)):
#             if all_features[idx] in keep_set:
#                 keep_idxs.append(idx)
#         return keep_idxs
        
    def filter_by_features(self, keep_features, mode='all'):
        if mode == 'all':
            all_features = self.data.feature_cols
        elif mode == 'perf':
            all_features = self.data.perf_feature_cols
        elif mode == 'plan':
            all_features = self.data.plan_feature_cols
            
        keep_set = set(keep_features)
        keep_idxs = []
        for idx in range(len(all_features)):
            if all_features[idx] in keep_set:
                keep_idxs.append(idx)
        return keep_idxs
    
    def filter_by_features_seperate(self, keep_features):
        keep_set = set(keep_features)
        plan_keep_idxs = []
        for idx in range(len(self.data.plan_feature_cols)):
            if self.data.plan_feature_cols[idx] in keep_set:
                plan_keep_idxs.append(idx)
        perf_keep_idxs = []      
        for idx in range(len(self.data.perf_feature_cols)):
            if self.data.perf_feature_cols[idx] in keep_set:
                perf_keep_idxs.append(idx)
        return plan_keep_idxs, perf_keep_idxs
    
    '''
    calculate matrix distances by DTW, only for resource utilization data (perf)
    '''
    def calc_dtw_simi_matrix(self, perf_feature_names=None, normalize=True):
        if perf_feature_names is None:
            perf_feature_names = self.data.perf_feature_cols
        keep_cols = self.filter_by_features(perf_feature_names, mode='perf')

        ndarrs = [b[:, keep_cols] for b in self.perf_mtxs_splitted]

        simi_mtx = np.zeros((len(ndarrs), len(ndarrs)))
        for i in range(len(ndarrs)): # for each experiment with idx i
            for j in range(i+1): # for each experiment idx <- i
                if i == j:
                    continue
                else:
                    simi_mtx[i][j] = self.__comp_dtw(ndarrs[i], ndarrs[j])
        for i in range(len(ndarrs)):
            for j in range(len(ndarrs)):
                if i < j:
                    simi_mtx[i][j] = simi_mtx[j][i]
        if normalize:
            self.simi_mtx = preprocessing.minmax_scale(simi_mtx.T).T
        else:
            self.simi_mtx = simi_mtx
        
    '''
    calculate timeseries matrix distances by Norms
    '''
    def calc_simi_matrix(self, perf_feature_names=None, norm_type='l11', normalize=True):
        if perf_feature_names is None:
            perf_feature_names = self.data.perf_feature_cols
        keep_cols = self.filter_by_features(perf_feature_names, mode='perf')
        
        min_length = np.min([b.shape[0] for b in self.perf_mtxs_splitted])

        ndarrs = [b[:min_length, keep_cols] for b in self.perf_mtxs_splitted]

        simi_mtx = np.zeros((len(ndarrs), len(ndarrs)))
        for i in range(len(ndarrs)): # for each experiment with idx i
            for j in range(i+1): # for each experiment idx <- i
                if i == j:
                    continue
                else:
                    if norm_type == 'l11':
                        simi_mtx[i][j] = self.__comp_l11_norm(ndarrs[i], ndarrs[j])
                    elif norm_type == 'l21':
                        simi_mtx[i][j] = self.__comp_l21_norm(ndarrs[i], ndarrs[j])
                    elif norm_type == 'fro':
                        simi_mtx[i][j] = self.__comp_fro_norm(ndarrs[i], ndarrs[j])
                    elif norm_type == 'canb':
                        simi_mtx[i][j] = self.__comp_canb_norm(ndarrs[i], ndarrs[j])
                    elif norm_type == 'chi2':
                        simi_mtx[i][j] = self.__comp_chi2_norm(ndarrs[i], ndarrs[j])
                    elif norm_type == 'corr':
                        simi_mtx[i][j] = self.__comp_corr_norm(ndarrs[i], ndarrs[j])
        for i in range(len(ndarrs)):
            for j in range(len(ndarrs)):
                if i < j:
                    simi_mtx[i][j] = simi_mtx[j][i]
        if normalize:
            self.simi_mtx = preprocessing.minmax_scale(simi_mtx.T).T
        else:
            self.simi_mtx = simi_mtx
        
    '''
    calculate matrix distances by DTW, only for resource utilization data (perf)
    '''
    def calc_ind_dtw_simi_matrix(self, perf_feature_names=None, normalize=True):
        if perf_feature_names is None:
            perf_feature_names = self.data.perf_feature_cols
        keep_cols = self.filter_by_features(perf_feature_names, mode='perf')

        ndarrs = [b[:, keep_cols] for b in self.perf_mtxs_splitted]

        simi_mtx = np.zeros((len(ndarrs), len(ndarrs)))
        for i in range(len(ndarrs)): # for each experiment with idx i
            for j in range(i+1): # for each experiment idx <- i
                if i == j:
                    continue
                else:
                    simi_mtx[i][j] = self.__comp_col_dtw(ndarrs[i], ndarrs[j])
        for i in range(len(ndarrs)):
            for j in range(len(ndarrs)):
                if i < j:
                    simi_mtx[i][j] = simi_mtx[j][i]
                    
        if normalize:
            self.simi_col_mtx = preprocessing.minmax_scale(simi_mtx.T).T
        else:
            self.simi_col_mtx = simi_mtx
    
    '''
    calculate matrix distances by LCSS, only for resource utilization data (perf)
    '''
    def calc_lcss_simi_matrix(self, perf_feature_names=None, normalize=True):
        if perf_feature_names is None:
            perf_feature_names = self.data.perf_feature_cols
        keep_cols = self.filter_by_features(perf_feature_names, mode='perf')

        ndarrs = [b[:, keep_cols] for b in self.perf_mtxs_splitted]

        simi_mtx = np.zeros((len(ndarrs), len(ndarrs)))
        for i in range(len(ndarrs)): # for each experiment with idx i
            for j in range(i+1): # for each experiment idx <- i
                if i == j:
                    continue
                else:
                    simi_mtx[i][j] = self.__comp_lcss(ndarrs[i], ndarrs[j])
        for i in range(len(ndarrs)):
            for j in range(len(ndarrs)):
                if i < j:
                    simi_mtx[i][j] = simi_mtx[j][i]
        if normalize:
            self.simi_mtx = preprocessing.minmax_scale(simi_mtx.T).T
        else:
            self.simi_mtx = simi_mtx
        
    '''
    calculate colwise matrix distances by LCSS, only for resource utilization data (perf)
    '''
    def calc_ind_lcss_simi_matrix(self, perf_feature_names=None, normalize=True):
        if perf_feature_names is None:
            perf_feature_names = self.data.perf_feature_cols
        keep_cols = self.filter_by_features(perf_feature_names, mode='perf')

        ndarrs = [b[:, keep_cols] for b in self.perf_mtxs_splitted]

        simi_mtx = np.zeros((len(ndarrs), len(ndarrs)))
        for i in range(len(ndarrs)): # for each experiment with idx i
            for j in range(i+1): # for each experiment idx <- i
                if i == j:
                    continue
                else:
                    simi_mtx[i][j] = self.__comp_col_lcss(ndarrs[i], ndarrs[j])
        for i in range(len(ndarrs)):
            for j in range(len(ndarrs)):
                if i < j:
                    simi_mtx[i][j] = simi_mtx[j][i]
        if normalize:
            self.simi_col_mtx = preprocessing.minmax_scale(simi_mtx.T).T
        else:
            self.simi_col_mtx = simi_mtx
        
    '''
    calculate matrix distances by Phase Distribution
    each phase is represented by  mean, median, variance, and augmented Dickey-Fuller test statistics
    '''
    def __phase_summarize(self, phase):
        if phase is None:
            return [0]*3
        # sample size is too short to use selected regression component
        # return [np.mean(phase), np.median(phase), np.var(phase), adfuller(phase)]
        return [np.mean(phase), np.median(phase), np.var(phase)]
     
    '''
    for 2 sets of feature phase summaries, compute their difference    
    '''
    def __comp_phase_dist(self, A, B, norm_type='l11'):
        distance = 0
        for phases_a, phases_b in zip(A, B):
            mtx_a, mtx_b = [], []
            # phases of a single feature
            max_phase = np.max([len(phases_a), len(phases_b)])
            for idx in range(max_phase):
                # wrap around
                a_idx = idx
                while a_idx >= len(phases_a):
                    a_idx -= len(phases_a)
                b_idx = idx
                while b_idx >= len(phases_b):
                    b_idx -= len(phases_b)
                curr_a = phases_a[a_idx]
                curr_b = phases_b[b_idx] 
                mtx_a.append(curr_a)
                mtx_b.append(curr_b)
            # now that mtx_a and mtx_b are both size max_phase * 4, where 4 is number of stats for each phase
            # compute Normalized euclidean for each stats (In the paper it also compute dtw as a comparison)
            #    Now that the stats are generated from normalized data, 
            #    but different stats (mean and variance for example) would have different magnitude
            #    Do a concate and scale for the two matrices and then calculate the euclidean distances
            scaler = ScaleData()
            splitted, _ = scaler.scale([np.array(mtx_a), np.array(mtx_b)])
            if norm_type == 'l11':
                distance += self.__comp_l11_norm(splitted[0], splitted[1])# /(1.0*max_phase)
            elif norm_type == 'l21':
                distance += self.__comp_l21_norm(splitted[0], splitted[1])# /(1.0*max_phase)
            elif norm_type == 'fro':
                distance += self.__comp_fro_norm(splitted[0], splitted[1])# /(1.0*max_phase)
            elif norm_type == 'canb':
                distance += self.__comp_canb_norm(splitted[0], splitted[1])# /(1.0*max_phase)
            elif norm_type == 'chi2':
                distance += self.__comp_chi2_norm(splitted[0], splitted[1])# /(1.0*max_phase)
            elif norm_type == 'corr':
                distance += self.__comp_corr_norm(splitted[0], splitted[1])# /(1.0*max_phase)
            # distance += self.__comp_l21_norm(splitted[0], splitted[1])# /(1.0*max_phase)
        return distance
                
        
    def calc_phase_simi_matrix(self, feature_names=None, cpd='Kernel', penalty=10, norm_type='l11', normalize=True):
        if feature_names is None:
            feature_names = self.data.feature_cols
        plan_keep_idxs, perf_keep_idxs = self.filter_by_features_seperate(feature_names)
        
        # initialize a empty list for each experiment run
        distributions = [[] for _ in self.plan_mtxs_splitted]
        
        # for plan features, summarize to one phase
        plan_ndarrs = [b[:, plan_keep_idxs] for b in self.plan_mtxs_splitted]
        for idx in range(len(plan_ndarrs)):
            # get the plan metrics for each experiment run
            plan_ndarr = plan_ndarrs[idx]
            # summarize each column (feature)
            for plan_col in plan_ndarr.T:
                # phase_summs = []
                phase0 = self.__phase_summarize(plan_col)
                distributions[idx].append([phase0])
        
        # for resource features, summarize to phases
        perf_ndarrs = [b[:, perf_keep_idxs] for b in self.perf_mtxs_splitted]
        for idx in range(len(perf_ndarrs)):
            perf_ndarr = perf_ndarrs[idx]
            # summarize each column
            for perf_col in perf_ndarr.T:
                phase_summs = []
                if cpd == 'Kernel':
                    algo = rpt.KernelCPD(kernel="rbf").fit(perf_col)
                elif cpd == 'bcpd':
                    algo = rpt.KernelCPD(kernel="rbf").fit(perf_col)

                result = algo.predict(pen=penalty)
                result = np.append(result, len(perf_col))
                prev = 0
                for end in result:
                    curr = perf_col[prev:end]
                    if end <= prev or len(curr) == 0:
                        continue
                    prev = end
                    phase = self.__phase_summarize(curr)
                    phase_summs.append(phase)
                distributions[idx].append(phase_summs)
                
        # for each run, zero-pad to the max number of phases
        for dist in distributions:
            phase_lengths = [len(uni_dist) for uni_dist in dist]
            max_num_phases = np.max(phase_lengths)
            for idx in range(len(dist)):
                for _ in range(max_num_phases - len(dist[idx])):
                    dist[idx].append(self.__phase_summarize(None))
                
        phase_simi_mtx = np.zeros((len(distributions), len(distributions)))
        for i in range(len(distributions)): # for each experiment with idx i
            for j in range(i+1): # for each experiment idx <- i
                if i == j:
                    continue
                else:
                    phase_simi_mtx[i][j] = self.__comp_phase_dist(distributions[i], distributions[j], norm_type)
        for i in range(len(distributions)):
            for j in range(len(distributions)):
                if i < j:
                    phase_simi_mtx[i][j] = phase_simi_mtx[j][i]
        if normalize:
            self.simi_mtx = preprocessing.minmax_scale(phase_simi_mtx.T).T
        else:
            self.simi_mtx = phase_simi_mtx
    
    '''
    Horizontally concatenate every element in plan_bined with perf_bined in a pair-wise fashion
    '''
    def __concate_mtxs(self):
        result = []
        for A, B in zip(self.plan_bined, self.perf_bined):
            if A is None:
                result.append(B)
            elif B is None:
                result.append(A)
            else:
                result.append(np.concatenate((A, B), axis=1))
        return result

    '''
    Given list of matrices, output the pairwise distribution similarity matrix
    '''
    def calc_dist_simi_matrix(self, cumulative=False, feature_names=None, norm_type='l11', normalize=True, timeit=False):
        if feature_names is None:
            feature_names = self.data.feature_cols
        keep_cols = self.filter_by_features(feature_names)
        
        if cumulative:
            bined = self.cumulative_bined
        else:
            bined = self.bined

        ndarrs = [b[:, keep_cols].astype(float) for b in bined]
        # print(ndarrs)
        if timeit:
            start = time.time()

        simi_mtx = np.zeros((len(ndarrs), len(ndarrs)))
        for i in range(len(ndarrs)): # for each experiment with idx i
            for j in range(i+1): # for each experiment idx <- i
                if i == j:
                    continue
                else:
                    if norm_type == 'l11':
                        simi_mtx[i][j] = self.__comp_l11_norm(ndarrs[i], ndarrs[j])
                    elif norm_type == 'l21':
                        simi_mtx[i][j] = self.__comp_l21_norm(ndarrs[i], ndarrs[j])
                    elif norm_type == 'fro':
                        simi_mtx[i][j] = self.__comp_fro_norm(ndarrs[i], ndarrs[j])
                    elif norm_type == 'canb':
                        simi_mtx[i][j] = self.__comp_canb_norm(ndarrs[i], ndarrs[j])
                    elif norm_type == 'chi2':
                        simi_mtx[i][j] = self.__comp_chi2_norm(ndarrs[i], ndarrs[j])
                    elif norm_type == 'corr':
                        simi_mtx[i][j] = self.__comp_corr_norm(ndarrs[i], ndarrs[j])
                            
        for i in range(len(ndarrs)):
            for j in range(len(ndarrs)):
                if i < j:
                    simi_mtx[i][j] = simi_mtx[j][i]
        if normalize:
            self.simi_mtx = preprocessing.minmax_scale(simi_mtx.T).T  
        else:
            self.simi_mtx = simi_mtx
            
        if timeit:
            end = time.time()
            print(f'Simi calculation time {end-start}')


    '''
    Given list of matrices, output the column wise pairwise distribution similarity matrix
    '''
    def calc_col_dist_simi_matrix(self, cumulative=False, feature_names=None, norm_type='l11', normalize=True):
        if feature_names is None:
            feature_names = self.data.feature_cols
        keep_cols = self.filter_by_features(feature_names)
        
        if cumulative:
            bined = self.cumulative_bined
        else:
            bined = self.bined

        ndarrs = [b[:, keep_cols].astype(float) for b in bined]

        simi_mtx = np.zeros((len(ndarrs), len(ndarrs)))
        for i in range(len(ndarrs)): # for each experiment with idx i
            for j in range(i+1): # for each experiment idx <- i
                if i == j:
                    continue
                else:
                    if norm_type == 'l11':
                        simi_mtx[i][j] = self.__comp_l11_norm(ndarrs[i], ndarrs[j])
                    elif norm_type == 'l21':
                        simi_mtx[i][j] = self.__comp_l21_norm(ndarrs[i], ndarrs[j])
                            
        for i in range(len(ndarrs)):
            for j in range(len(ndarrs)):
                if i < j:
                    simi_mtx[i][j] = simi_mtx[j][i]
        if normalize:
            self.simi_col_mtx = preprocessing.minmax_scale(simi_mtx.T).T
        else:
            self.simi_col_mtx = simi_mtx


    '''
    Given list of matrices, output the pairwise distribution similarity matrix
    '''
    def calc_featurewise_dist_by_col(self, feature_names=None):
        if feature_names is None:
            feature_names = self.data.feature_cols        
        bined = [b[:, self.filter_by_features(feature_names)] for b in self.bined]
        
        # #exprs * #exprs * #features
        simi_col_mtx = []
        
        # for each expr A
        for i in range(len(bined)):
            # hist_mtx: #bins * #features
            base_hist_mtx = bined[i]
            # featurewise_simi_mtx: #exprs * #features
            featurewise_simi_mtx = []
            
            # each expr B
            for j in range(i+1):
                if i == j:
                    curr_dist_list = np.zeros(len(feature_names))
                else:
                    hist_mtx = bined[j]
                    curr_dist_list = self.__comp_col_distribution_fp(base_hist_mtx, hist_mtx)
                featurewise_simi_mtx.append(curr_dist_list)
            simi_col_mtx.append(featurewise_simi_mtx)
        
        for i in range(len(bined)):
            for j in range(len(bined)):
                if i < j:
                    assert(len(simi_col_mtx[i]) == j)
                    simi_col_mtx[i].append(simi_col_mtx[j][i])

        self.simi_col_mtx = np.array(simi_col_mtx)
        print(self.simi_col_mtx.shape)


# In[7]:


@update_class()
class Similarity():
    '''
    Generate scatter plot by similarity
    Attributes:
        simi: List of distances; used as x-axis. 
              This shoud be the i-th row/column from the similarity matrix, 
              representing the similarity distance to the workload representation of the i-th experiment run. 
        wl_name: Type of the workload (tpcc, tpch, twitter) of the experiment used as center.
        expr_idx: Index of the experiment used as center.
        cpu_num_value: Number of cpus of the experiment used as center
        data_type: Type of data used to calculate similarity (plan, event)
        method_type: Type of method used to calculate similarity
    '''
    def scatter_plot(self, expr_idx, data_type, method_type, dir_name, note=''):
        simi = self.simi_mtx[expr_idx]
        wl_name = self.data.wl_names[expr_idx]
        group_idx = self.data.wl_groups[expr_idx]
        cpu_num_val = self.data.cpu_nums[expr_idx]
        zipped = list(zip(self.data.wl_groups, self.data.wl_names, self.data.cpu_nums, simi))
        X_p = pd.DataFrame(zipped, columns=['expr', 'workload', 'cpu_num', 'distance'])
        X_p = X_p.astype({'distance': float})

        fig, ax = plt.subplots()

        sns.stripplot(x="distance", y="cpu_num", hue="workload", data=X_p, ax=ax, palette=sns.color_palette()[:X_p['workload'].nunique()])
        # sns.scatterplot(x="distance", y="cpu_num", hue="workload", data=X_p, ax=ax, palette=sns.color_palette()[:X_p['workload'].nunique()])
        title = 'Workload: {}, Experiment: {}, CPU: {}.{}'.format(wl_name, group_idx, cpu_num_val, note)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

        plt.title(title)
        if dir_name is not None:
            fname = 'new{}{}_cpu{}_{}_{}{}.pdf'.format(wl_name, group_idx, cpu_num_val, data_type, method_type, note)
            path = os.path.join(dir_name, fname)
            plt.savefig(path)
        plt.show()


# # simi_penalty function (only used for feature study)
# 
# Try to formalize it.

# In[8]:


@update_class()
class Similarity():
    def __majority_label_vote(self, labels):
        freqDict = Counter(labels)
        max_count = 0
        max_label = None
        for l in labels:
            val = freqDict[l]
            if val > max_count:
                max_label = l
                max_count = val
        return max_label

    # knn distance scoring?
    # if label wrong: 10
    # else cpu_num_diff*0.1
    def simi_penalty(self, n=1, dependent=True):
        if dependent:
            simi = self.simi_mtx
        else:
            simi = self.simi_col_mtx
        wrong_penalty = 10
        param_penalty_rate = 1

        penaltys = []
        # knn predicting the 
        for i in range(simi.shape[0]):
            sort_index = np.argsort(simi[i])[1:n+1]
            # print(sort_index)
            # print(np.argsort(simi[i]))
            labels = [self.data.wl_names[j] for j in sort_index]
            pred_label = self.__majority_label_vote(labels)
            if pred_label != self.data.wl_names[i]:
                penaltys.append(wrong_penalty)
                # print("wrong pred", self.data.wl_names[i], self.data.wl_groups[i])
            else:
                # secs = [self.data.cpu_nums[k] for k in sort_index]
                secs = [self.data.terminal_num[k] for k in sort_index]
                
                pred_sec = self.__majority_label_vote(secs)
                # if pred_sec != self.data.cpu_nums[i]:
                if pred_sec != self.data.terminal_num[i]:
                    penaltys.append(param_penalty_rate)
                else:
                    penaltys.append(0)
        return np.mean(penaltys), penaltys
    
    def simi_pred(self, n=1, dependent=True):
        if dependent:
            simi = self.simi_mtx
        else:
            simi = self.simi_col_mtx

        pred_labels = []
        # knn predicting 
        for i in range(simi.shape[0]):
            sort_index = np.argsort(simi[i])[1:n+1]
            labels = [self.data.wl_names[j] for j in sort_index]
            pred_label = self.__majority_label_vote(labels)
            pred_labels.append(pred_label)
        return pred_labels

# In[ ]:




