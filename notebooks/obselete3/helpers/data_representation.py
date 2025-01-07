#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
import functools

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


# ## DataRepresentation Computation
# 
# For each type of data (plan, event), we try to calculate DataRepresentation based on:
# 
# 1. bined distribution of each feature;
# 2. shape and absolute values of timeseries data

# In[3]:


class DataRepresentation():
    def __init__(self, data, plan_mtxs_splitted, plan_col_ranges, perf_mtxs_splitted, perf_col_ranges, num_bins=10):
        self.plan_mtxs_splitted = plan_mtxs_splitted
        self.plan_col_ranges = plan_col_ranges
        self.perf_mtxs_splitted = perf_mtxs_splitted
        self.perf_col_ranges = perf_col_ranges
        self.col_ranges = plan_col_ranges+perf_col_ranges
        
        self.num_bins = num_bins
        self.plan_bined = None
        self.plan_dist_mtx = None
        self.perf_bined = None
        self.perf_dist_mtx = None # Not used
        
        self.bined = None
        self.dist_mtx = None
        
        self.plan_dist_by_col_cube = None
        self.dist_by_col_cube = None

        self.data = data


# In[4]:


@update_class()
class DataRepresentation():
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
class DataRepresentation():
    '''
    Calculate frequency based bins (each bin value range from 0 to 10 and sums to 10)
    '''
    def calc_bined_mtx(self):
        self.plan_bined = []
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
        self.bined = self.__concate_mtxs()
        # self.col_ranges = np.append(self.plan_col_ranges, self.perf_col_ranges)


# In[6]:


@update_class()
class DataRepresentation():
    '''
    for 2 matrices, compare their distributions
    '''
    def __comp_dist(self, A, B):
        return np.absolute(A - B).sum()
    '''
    for 2 matrices, compare their column-wise distributions
    '''
    def __comp_col_dist(self, A, B):
        return np.sum(np.absolute(A - B), axis=0)
    
    
    '''
    filter by features
    '''
    def __filter_by_features(self, keep_features):
        all_features = self.data.feature_cols
        keep_set = set(keep_features)
        keep_idxs = []
        for idx in range(len(all_features)):
            if all_features[idx] in keep_set:
                keep_idxs.append(idx)
        return keep_idxs
    
    '''
    Concatenate every element in plan_bined with perf_bined in a pair-wise fashion
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
    Given list of matrices, output the pairwise distribution DataRepresentation matrix
    '''
    def cal_dist_simi_matrix(self, feature_names=None):
        if feature_names is None:
            feature_names = self.data.feature_cols
        keep_cols = self.__filter_by_features(feature_names)
            
        ndarrs = [b[:, keep_cols] for b in self.bined]

        dist_mtx = np.zeros((len(ndarrs), len(ndarrs)))
        for i in range(len(ndarrs)): # for each experiment with idx i
            for j in range(i+1): # for each experiment idx <- i
                if i == j:
                    continue
                else:
                    dist_mtx[i][j] = self.__comp_dist(ndarrs[i], ndarrs[j])
        for i in range(len(ndarrs)):
            for j in range(len(ndarrs)):
                if i < j:
                    dist_mtx[i][j] = dist_mtx[j][i]
        self.dist_mtx = dist_mtx

    '''
    Given list of matrices, output the pairwise distribution DataRepresentation matrix
    '''
    def cal_featurewise_dist_by_col(self, feature_names=None):
        if feature_names is None:
            feature_names = self.data.feature_cols
        bined = [b[:, self.__filter_by_features(feature_names)] for b in self.bined]
        # #exprs * #exprs * #features
        dist_by_col_cube = []
        
        # for each expr A
        for i in range(len(bined)):
            # hist_mtx: #bins * #features
            base_hist_mtx = bined[i]
            # featurewise_dist_mtx: #exprs * #features
            featurewise_dist_mtx = []
            
            # each expr B
            for j in range(i+1):
                if i == j:
                    curr_dist_list = np.zeros(len(feature_names))
                else:
                    hist_mtx = bined[j]
                    curr_dist_list = self.__comp_col_dist(base_hist_mtx, hist_mtx)
                featurewise_dist_mtx.append(curr_dist_list)
            dist_by_col_cube.append(featurewise_dist_mtx)
        
        for i in range(len(bined)):
            for j in range(len(bined)):
                if i < j:
                    assert(len(dist_by_col_cube[i]) == j)
                    dist_by_col_cube[i].append(dist_by_col_cube[j][i])

        self.dist_by_col_cube = np.array(dist_by_col_cube)


# In[7]:


@update_class()
class DataRepresentation():
    '''
    Generate scatter plot by DataRepresentation
    Attributes:
        simi: List of distances; used as x-axis. 
              This shoud be the i-th row/column from the DataRepresentation matrix, 
              representing the DataRepresentation distance to the workload representation of the i-th experiment run. 
        wl_name: Type of the workload (tpcc, tpch, twitter) of the experiment used as center.
        expr_idx: Index of the experiment used as center.
        cpu_num_value: Number of cpus of the experiment used as center
        data_type: Type of data used to calculate DataRepresentation (plan, event)
        method_type: Type of method used to calculate DataRepresentation
    '''
    def scatter_plot(self, expr_idx, data_type, method_type, dir_name, note=''):
        simi = self.dist_mtx[expr_idx]
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
class DataRepresentation():
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
    def simi_penalty(self, n=5):
        simi = self.dist_mtx
        wrong_penalty = 10
        param_penalty_rate = 0.1

        penaltys = []
        # knn predicting the 
        for i in range(simi.shape[0]):
            sort_index = np.argsort(simi[i])[1:n+1]
            labels = [self.data.wl_names[j] for j in sort_index]
            pred_label = self.__majority_label_vote(labels)
            if pred_label != self.data.wl_names[i]:
                # print(i, pred_label)
                penaltys.append(wrong_penalty)
                # print("wrong pred", self.data.wl_names[i], self.data.wl_groups[i])
            else:
                # get the average cpu num of majority
                # cpu_num_sum = 0
                # for l, ind in zip(labels, sort_index):
                #     if l == pred_label:
                #         cpu_num_sum += self.data.cpu_nums[ind]
                # diff = abs(cpu_num_sum/n - self.data.cpu_nums[i])
                # penaltys.append(diff * param_penalty_rate)
                skus = [self.data.cpu_nums[k] for k in sort_index]
                pred_sku = self.__majority_label_vote(skus)
                if pred_sku != self.data.cpu_nums[i]:
                    penaltys.append(param_penalty_rate)
                else:
                    penaltys.append(0)
        return np.mean(penaltys), penaltys


# In[ ]:




