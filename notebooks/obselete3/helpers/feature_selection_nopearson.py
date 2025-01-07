import numpy as np
import pandas as pd
from itertools import compress
import time

from sklearn.linear_model import lasso_path, enet_path, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR, LinearSVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
from sklearn.feature_selection import VarianceThreshold, SelectKBest, RFE, SequentialFeatureSelector, SelectFromModel
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif

from boruta import BorutaPy
import scipy.stats as ss

from helpers.expr_data_pickle5 import ExprDataPickle5 as ExprData
from helpers.similarity_nophase import SimilarityNoPhase as Similarity

import warnings
warnings.filterwarnings("ignore")

class FeatureSelectionNoPearson():
    def __init__(self, simi_calc):
        self.simi_calc = simi_calc
        
    # return non-zero index in descending order
    @staticmethod
    def __sparse_argsort(arr):
        arr = np.where(np.isnan(arr), 0, arr)
        arr = arr * -1
        indices = np.nonzero(arr)[0]
        result = indices[np.argsort(arr[indices])]
        return result
    
    @staticmethod
    def __all_argsort(arr):
        arr = np.where(np.isnan(arr), 0, arr)
        arr = arr * -1
        result = np.argsort(arr)
        return result
        
    @staticmethod
    def __variance_threshold(X, y):
        selector = VarianceThreshold()
        selector.fit(X)
        scores = selector.variances_
        return FeatureSelectionNoPearson.__sparse_argsort(scores)
    
    
    # fANOVA, Chi-Squared test, Mutual Information gain
    @staticmethod
    def __select_k_best(X, y, method):
        if method == 'fANOVA':
            selector = SelectKBest(f_classif, k='all')
        elif method == 'Chi2':
            selector = SelectKBest(chi2, k='all')
        elif method == 'MutualInfoGain': # this uses knn=3 by default to do the feature selection 
            selector = SelectKBest(mutual_info_classif, k='all')  
        selector.fit(X, y)
        scores = np.absolute(selector.scores_)
        return FeatureSelectionNoPearson.__sparse_argsort(scores)
    
    @staticmethod
    def __get_est(est_name):
        if est_name == 'DecisionTree':
            estimator = DecisionTreeClassifier(criterion='entropy', max_depth=None)
        elif est_name == 'LogisticRegression':
            estimator = LogisticRegression(n_jobs=4, C=0.01) # C, tol, 
        else: # est_name == 'Linear':
            estimator = SVR(kernel="linear", C=0.05, ) # kernel, degree, 
        return estimator
    
    @staticmethod
    # create customize base estimator: https://stackoverflow.com/questions/51679173/using-sklearn-rfe-with-an-estimator-from-another-package
    def __rfe_orders(X, y, est_name):
        estimator = get_est(est_name)
        selector = RFE(estimator, n_features_to_select=1, step=1)
        selector = selector.fit(X, y)
        # ranks are: 1 for most important
        scores = -1 * selector.ranking_
        return FeatureSelectionNoPearson.__sparse_argsort(scores)
    
    @staticmethod
    def __lasso_weights_orders(X, y):
        selector = SelectFromModel(estimator=LinearSVC(C=0.01, penalty="l1", dual=False)).fit(X, y) # C 
        scores = np.abs(selector.estimator_.coef_[0])
        return FeatureSelectionNoPearson.__sparse_argsort(scores)

    @staticmethod
    def __ridge_weights_orders(X, y):
        selector = SelectFromModel(estimator=LinearSVC(C=0.01, penalty="l2", dual=False)).fit(X, y)
        scores = np.abs(selector.estimator_.coef_[0])
        return FeatureSelectionNoPearson.__sparse_argsort(scores)
    
    @staticmethod
    def boruta_orders(X, y, est_name):
        boruta = BorutaPy(estimator=RandomForestClassifier(n_estimators=10, n_jobs=6), n_estimators=10, max_iter=5)
        # Fits Boruta
        boruta.fit(X, y)
        scores = -1 * boruta.ranking_
        return FeatureSelectionNoPearson.__sparse_argsort(scores)
    
    def __get_candidate_feature_cols(self, feature_type):
        if feature_type is None:
            feature_cols = self.simi_calc.data.feature_cols
        elif feature_type == 'plan':
            feature_cols = self.simi_calc.data.plan_feature_cols
        elif feature_type == 'perf':
            feature_cols = self.simi_calc.data.perf_feature_cols
        else:
            print("Invalid feature type")
            return None
        return feature_cols
        
    def __get_feature_orders(self, method, est_name=None, feature_type=None):
        feature_cols = self.__get_candidate_feature_cols(feature_type)
        
        num_features = len(feature_cols)
        feature_importance = np.array([0]*num_features)
        expr_num = self.simi_calc.data.get_num_exprs()
        
        self.simi_calc.calc_featurewise_dist_by_col(feature_cols)

        for i in range(expr_num):
            # calculate label
            curr_name = self.simi_calc.data.wl_names[i]
            y = [curr_name == name for name in self.simi_calc.data.wl_names]
            X = self.simi_calc.simi_col_mtx[i]
            mask = np.ones(X.shape[0], dtype=bool)  
            X = X[mask]

            if method == 'Lasso':
                orders = FeatureSelectionNoPearson.__lasso_weights_orders(X, y)
            elif method == 'Ridge':
                orders = FeatureSelectionNoPearson.__ridge_weights_orders(X, y)
            elif method == 'Variance':
                orders = FeatureSelectionNoPearson.__variance_threshold(X, y)
            elif method == 'fANOVA':
                orders = FeatureSelectionNoPearson.__select_k_best(X, y, method='fANOVA')
            elif method == 'Chi2':
                orders = FeatureSelectionNoPearson.__select_k_best(X, y, method='Chi2')
            elif method == 'MutualInfoGain':
                orders = FeatureSelectionNoPearson.__select_k_best(X, y, method='MutualInfoGain')
            elif method == 'Pearson':
                orders = FeatureSelectionNoPearson.__select_k_best(X, y, method='Pearson')
            elif method == 'RFE':
                orders = FeatureSelectionNoPearson.__rfe_orders(X, y, est_name)
            elif method == 'Boruta':            
                orders = FeatureSelectionNoPearson.__boruta_orders(X, y, est_name)

            for idx in range(len(orders)):
                # from 0 to last idx of orders
                # the score = num_features - idx
                #   for a entry with feature_idx important order idx idx
                # the higher the order, the more the score
                feature_importance[orders[idx]] += num_features-idx

        final_orders = FeatureSelectionNoPearson.__sparse_argsort(feature_importance)
        # final_orders = FeatureSelectionNoPearson.__all_argsort(feature_importance)
        top_features = [feature_cols[j] for j in final_orders]
        return top_features
    
    def __sfs_orders(self, num_features, direction, est_name, feature_type=None):
        feature_cols = self.__get_candidate_feature_cols(feature_type)

        num_features = len(self.simi_calc.feature_cols)
        feature_importance = np.array([0]*num_features)
        expr_num = self.simi_calc.data.get_num_exprs()

        self.simi_calc.calc_featurewise_dist_by_col(feature_cols)

        for i in range(expr_num):
            # calculate label
            curr_name = self.simi_calc.wl_names[i]
            y = [curr_name == name for name in self.simi_calc.wl_names]
            X = simi_calc.simi_col_mtx[i]
            estimator = get_est(est_name)

            selector = SequentialFeatureSelector(estimator, direction=direction.lower(), n_features_to_select=num_features, n_jobs=-2, cv=3)
            selector = selector.fit(X, y)
            mask = selector.get_support()
            for idx in range(num_features):
                feature_importance[idx] += mask[idx]
        final_orders = FeatureSelectionNoPearson.__sparse_argsort(feature_importance)[:num_features]
        top_features = [feature_cols[j] for j in final_orders]
        return top_features
    
        
    '''
    num_features: Number of most important features to consider
    fs_method: Feature selection method used for ordering feature importance or selecting features.
               Valid fs_methods are: 
    est_name: Name of underlying estimator if using wrapper-based feature selection method
    direction: Forward or Backward wrapper type feature selection; only for SFS
    '''
    def select_features(self, num_features, fs_method, est_name=None, direction=None, feature_type=None):
        if fs_method == 'SFS':
            f_features = self.__sfs_orders(num_features, direction, est_name, feature_type)
        else:
            top_features = self.__get_feature_orders(fs_method, est_name, feature_type)
            f_features = top_features[:num_features]
        return f_features

    
    '''
    num_features: Number of most important features to consider
    fs_method: Feature selection method used for ordering feature importance or selecting features.
               Valid fs_methods are: 
    est_name: Name of underlying estimator if using wrapper-based feature selection method
    knn_threshold: Number of nearest neighbor used for KNN similarity penalty calculation
    num_repeats: Number of experiments repeats (the final feature importance ordering is the average of experiments)
    '''
    def calc_feature_selection_accuracy(self, num_features, fs_method, est_name=None, knn_threshold=1, num_repeats=10):
        accs = []
        times = []
        # run n times to get the average
        for i in range(num_repeats):       
            start_time = time.time()
            top_features = self.__get_feature_orders(fs_method, est_name)
            times_elapsed = time.time() - start_time
            # print("fs --- %s seconds ---" % times_elapsed)
            times.append(times_elapsed)
            f_features = top_features[:num_features]

            self.simi_calc.cal_dist_simi_matrix(f_features)
            pen, pens = self.simi_calc.simi_penalty(n=knn_threshold)
            acc = 1 - (np.sum(pens)/(len(pens)*10))
            accs.append(curr_accs)
        acc_result = np.average(np.array(accs))
        time_result = np.average(np.array(times))
        
        return acc_result, time_result
        
