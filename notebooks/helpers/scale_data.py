#!/usr/bin/env python
# coding: utf-8

# # Scale Data
# 
# We concatenate all dataframes of the same type (plan or event) together, then apply a desired scaler on each column.
# Dimensionality reduction may also be done along the way.
# After scaling, we split the dataframes back to desired sizes

# In[1]:


import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

import functools


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


# In[3]:


class ScaleData():
    def __init__(self):
        self.__pkl_fdn = '../model/processed_wl/'


# In[4]:


@update_class()
class ScaleData():
    # X should have been properly scaled before calling this method
    def __pca(X):
        # sc = RobustScaler()
        # X = sc.fit_transform(X)
        pca = PCA(n_components=1)
        X_pca = pca.fit_transform(X)
        return X_pca.reshape(1, -1)[0]


# In[5]:


@update_class()
class ScaleData():
    # in the list of dataframes, get the minimum number of rows for a dataframe
    def get_min_size(dfs):
        min_size = dfs[0].shape[0]
        for i in range(len(dfs)):
            curr_size = dfs[0].shape[0]
            if curr_size < min_size:
                min_size = curr_size
        return min_size


# In[6]:


@update_class()
class ScaleData():
    '''
    Attributes:
        ndarrs: list of ndarries
        fixed_size: number of rows to keep in each ndarray; keep all if None
    Return:
        temp_np: one concatenated ndarray
    '''
    def __concate_mtx(self, ndarrs, fixed_size=None):
        # made them same length
        if fixed_size is None:
            temp_np = np.array(ndarrs[0])
        else:
            temp_np = np.array(ndarrs[0][:fixed_size])
        for i in range(len(ndarrs)):
            if ndarrs[i] is not None and i > 0:
                curr_size = ndarrs[i].shape[0] if fixed_size is None else fixed_size
                temp_np = np.append(temp_np, ndarrs[i][:curr_size], axis=0)
        return temp_np

    '''
    Attributes:
        temp_np: one ndarray
        scaler: the scaler object
        pca: if PCA is performed on temp_np to collapse to one column
    Return:
        after_scale: the ndarray after scaling and PCA transform if applied
        col_ranges: list of feature-wise min-max tuple after scaling
    '''
    def __scale_mtxs(self, temp_np, scaler, use_pca=False):
        after_scale = scaler.fit_transform(temp_np)

        if use_pca:
            after_scale = self.__pca(after_scale).reshape(-1, 1)
        col_ranges = []
        for i in range(after_scale.shape[1]):
            curr_list = after_scale[ : , i]
            col_ranges.append((np.min(curr_list), np.max(curr_list)))
        return after_scale, col_ranges

    '''
    Attributes:
        temp_np: one concatenated ndarray
        ndarrs: the list of ndarries before concatenation
        fixed_size: number of rows to keep in each splited ndarray; keep origin if None
    Return:
        std_arrs: list of ndarries after scaling
    '''
    def __split_mtx(self, temp_np, ndarrs, fixed_size=None):
        std_arrs = []

        start = 0
        for mtx in ndarrs:
            if mtx is None:
                std_arrs.append(None)
            else:
                curr_size = mtx.shape[0] if fixed_size is None else fixed_size
                std_arrs.append(temp_np[start: start+curr_size])
                start = start + curr_size
        return std_arrs


# In[7]:


@update_class()
class ScaleData():
    def scale(self, plan_mtxs):
        if plan_mtxs is None:
            return
        plan_con_all = self.__concate_mtx(plan_mtxs)
        scaler = MinMaxScaler()
        plan_mtxs_con_scaled, col_ranges = self.__scale_mtxs(plan_con_all, scaler)
        plan_mtxs_splitted = self.__split_mtx(plan_mtxs_con_scaled, plan_mtxs)
        return plan_mtxs_splitted, col_ranges

