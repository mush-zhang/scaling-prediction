#!/usr/bin/env python
# coding: utf-8

# # Load data
# 
# User should call `load_pickle()` or `process_csv_and_load()` to load data, not both.
# 
# If there are preprocessed pickle files in `model/processed_wl`, use `load_pickle()` for fast loading.
# 
# If the data has not been preprocessed, use `process_csv_and_load()` once to process the CSV files, save the processed dataframes to pickle files, and load the data to class variables. User can access the data afterwards with `load_pickle()`.
# 
# ##### Public Variables
# 
# Except for `feature_cols`, public variables are lists of the same length - #workloads * #SKU_configs * #experiment_sets. Elements of the same index corresponds to the same experiment run.
# 
# - wl_groups: list of experiment set indexes for all experiment runs.
# - wl_names: list of workload names for all experiment runs.
# - cpu_nums: list of number of CPUs (SKU configurations) for all experiment runs.
# - query_event_dfs: list of query event dataframes for all experiment runs. 
# - query_plan_dfs: list of processed query plan dataframes (numerical values only) for all experiment runs. Usually use `plan_mtxs` instead of this variable.
# - plan_mtxs: list of query event numpy arrays for all experiment runs.
# - feature_cols: Features of plans used. Length of the array equals to the number of columns of each element in `query_plan_dfs` and `plan_mtxs`.

# In[1]:


import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

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


class ExprData2():
    def __init__(self, 
                 wl_groups=None, wl_names=None, cpu_nums=None, 
                 query_event_dfs=None, query_plan_dfs=None, query_perf_dfs=None,
                 plan_mtxs=None, perf_mtxs=None, plan_feature_cols=None, perf_feature_cols=None):
        self.__pkl_fdn = '../model/processed_wl/hyperscale2'
        self.__fdn = '../model/hyperscale/'
        self.__wl_group_prefix = 'workload_'
        self.__config_prefix = 'cpu'
        self.__names = ['tpcc', 'tpch', 'twitter', 'twitter', 'tpch', 'tpch', 'tpcc', 'tpcc', 'twitter']
        self.wl_groups = wl_groups
        self.wl_names = wl_names
        self.cpu_nums = cpu_nums
        self.query_event_dfs = query_event_dfs
        self.query_perf_dfs = query_perf_dfs
        self.query_plan_dfs = query_plan_dfs
        self.plan_mtxs = plan_mtxs
        self.perf_mtxs = perf_mtxs
        self.wl_throughput = None
        self.wl_latency = None
        self.terminal_num = None
        self.wl_throughput_samples = None
        self.wl_latency_samples = None
        if plan_feature_cols is not None:
            self.plan_feature_cols = plan_feature_cols
        else:
            self.plan_feature_cols = ['StatementEstRows', 'StatementSubTreeCost', 'CachedPlanSize',
                                       'CompileCPU', 'CompileMemory', 'CompileTime', 'GrantedMemory',
                                       'MaxUsedMemory', 'SerialDesiredMemory', 'SerialRequiredMemory',
                                       'EstimatedAvailableDegreeOfParallelism',
                                       'EstimatedAvailableMemoryGrant', 'EstimatedPagesCached',
                                       'MaxCompileMemory', 'AvgRowSize', 'EstimateCPU', 'EstimateIO',
                                       'EstimateRebinds', 'EstimateRewinds', 'EstimateRows',
                                       'TableCardinality', 'EstimatedRowsRead']
        if perf_feature_cols is not None:
            self.perf_feature_cols = perf_feature_cols
        else:
            self.perf_feature_cols = ['CPU_UTILIZATION', 'CPU_EFFECTIVE', 'MEM_UTILIZATION', 'IOPS_TOTAL', 'READ_WRITE_RATIO', 'LOCK_REQ_ABS', 'LOCK_WAIT_ABS']

        self.feature_cols = np.concatenate((plan_feature_cols, perf_feature_cols), axis=None)


# In[4]:


@update_class()
class ExprData2():
    def get_num_exprs(self):
        return 0 if self.wl_groups is None else len(self.wl_groups)


# ### Functions to Parse Query Plans to Matrices
# 
# Helper functions to process query plan CSV files and extract the numerical values in plans to dataframes

# In[5]:


@update_class()
class ExprData2():
    def __parse_query_to_tree(self, plan):
        root = ET.fromstring(plan)
        xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent='\t')
        return root, xmlstr
    
    def __extract_next(self, node, tag):
        ns = '{http://schemas.microsoft.com/sqlserver/2004/07/showplan}'
        curr = node.findall(ns+tag)
        if curr is None or len(curr) == 0:
            return None
        return curr[0]
    
    def __extract_value(self, node, tag):
        if node is None or tag not in node.attrib:
            return -1
        return float(node.attrib[tag])

    def __extract_fv_from_tree(self, root):
        df = dict({})
        curr = self.__extract_next(root, 'BatchSequence')
        curr = self.__extract_next(curr, 'Batch')
        curr = self.__extract_next(curr, 'Statements')
        curr = self.__extract_next(curr, 'StmtSimple')
        df['StatementEstRows'] = self.__extract_value(curr, 'StatementEstRows')
        df['StatementSubTreeCost'] = self.__extract_value(curr, 'StatementSubTreeCost')
        curr = self.__extract_next(curr, 'QueryPlan')
        df['CachedPlanSize'] = self.__extract_value(curr, 'CachedPlanSize')
        df['CompileCPU'] = self.__extract_value(curr, 'CompileCPU')
        df['CompileMemory'] = self.__extract_value(curr, 'CompileMemory')
        df['CompileTime'] = self.__extract_value(curr, 'CompileTime')
        child1 = self.__extract_next(curr, 'MemoryGrantInfo')
        df['GrantedMemory'] = self.__extract_value(child1, 'GrantedMemory')
        df['MaxUsedMemory'] = self.__extract_value(child1, 'MaxUsedMemory')
        df['SerialDesiredMemory'] = self.__extract_value(child1, 'SerialDesiredMemory')
        df['SerialRequiredMemory'] = self.__extract_value(child1, 'SerialRequiredMemory')
        child2 = self.__extract_next(curr, 'OptimizerHardwareDependentProperties')
        df['EstimatedAvailableDegreeOfParallelism'] = self.__extract_value(child2, 'EstimatedAvailableDegreeOfParallelism')
        df['EstimatedAvailableMemoryGrant'] = self.__extract_value(child2, 'EstimatedAvailableMemoryGrant')
        df['EstimatedPagesCached'] = self.__extract_value(child2, 'EstimatedPagesCached')
        df['MaxCompileMemory'] = self.__extract_value(child2, 'MaxCompileMemory')
        child3 = self.__extract_next(curr, 'RelOp')
        df['AvgRowSize'] = self.__extract_value(child3, 'AvgRowSize')
        df['EstimateCPU'] = self.__extract_value(child3, 'EstimateCPU')
        df['EstimateIO'] = self.__extract_value(child3, 'EstimateIO')
        df['EstimateRebinds'] = self.__extract_value(child3, 'EstimateRebinds')
        df['EstimateRewinds'] = self.__extract_value(child3, 'EstimateRewinds')
        df['EstimateRows'] = self.__extract_value(child3, 'EstimateRows')
        # df['EstimatedRowsRead'] = self.__extract_value(child3, 'EstimatedRowsRead')
        df['TableCardinality'] = self.__extract_value(child3, 'TableCardinality')
        df['EstimatedRowsRead'] = self.__extract_value(child3, 'EstimatedRowsRead')
        return df


# In[6]:


@update_class()
class ExprData2():
    '''
    Attributes:
        query_plan_dfs: dataframe of query info from a workload, one column is query plan
    return: 
        mtxs: list of ndarraies where for each ndarray (mtx), 
              each row is plan's top level numerical feature, and each row is a query plan (not distinct)
    '''
    def __plan_to_mtx(self):
        mtxs = []
        temp_df = pd.DataFrame()
        fea_col_filled = False
        self.plan_feature_cols = None
        for i in range(len(self.query_plan_dfs)):
            # print(self.query_plan_dfs[i]['PLAN_HANDLE'].duplicated().any())
            fv_df = pd.DataFrame()
            for idx, row in self.query_plan_dfs[i].iterrows():
                curr_p = row.QUERY_PLAN
                plan_tree, plan_string = self.__parse_query_to_tree(curr_p)
                curr_dict = self.__extract_fv_from_tree(plan_tree)
                curr_df = pd.DataFrame([curr_dict])
                fv_df = pd.concat([fv_df, curr_df], axis=0, ignore_index=True)
                if not fea_col_filled:
                    self.plan_feature_cols = fv_df.columns
                    fea_col_filled = True
            mtxs.append(fv_df)
        self.query_plan_dfs = mtxs
        
    def __stream_plan_to_mtx(self, filename, fv_df):
        fea_col_filled = False
        line_num = 0
        with open(filename) as f:
            for plan_string in f:
                if line_num % 100000 == 0:
                    print(f'-- reading the No. {line_num} plan')
                line_num += 1
                fv_df = self.__indi_plan_to_mtx(fv_df, plan_string, fea_col_filled)
                if not fea_col_filled:
                    fea_col_filled = True
        return fv_df
        
    def __indi_plan_to_mtx(self, fv_df, plan_string, fea_col_filled):
        plan_tree, plan_string = self.__parse_query_to_tree(plan_string)
        curr_dict = self.__extract_fv_from_tree(plan_tree)
        curr_df = pd.DataFrame([curr_dict])
        fv_df = pd.concat([fv_df, curr_df], axis=0, ignore_index=True)
        if not fea_col_filled:
            self.plan_feature_cols = fv_df.columns
        return fv_df


# In[7]:


@update_class()
class ExprData2():
    def __process_csv_to_pickle(self):
        '''
        lists of:
        1. workload number (1-9)
        2. workload type (tpcc, tpch, twitter)
        3. number of cpu
        4. dataframe from query_event_log.csv
        4. dataframe from query_perf_log.csv
        5. dataframe from query_info_log.csv
        '''

        mode = 0o666

        for expr_num, wl_type, cpu_num, qe_df, perf_df, plan_df in zip(self.wl_groups, self.wl_names, self.cpu_nums, self.query_event_dfs, self.query_perf_dfs, self.query_plan_dfs):
            new_fname_prefix = '_'.join([wl_type, str(expr_num), str(cpu_num)])
            new_dir = os.path.join(self.__pkl_fdn, new_fname_prefix)
            try:
                os.mkdir(new_dir, mode)
            except:
                pass
            if qe_df is not None:
                qe_df.to_pickle(os.path.join(new_dir, 'query_event.pkl'))
            if perf_df is not None:
                perf_df.to_pickle(os.path.join(new_dir, 'query_perf.pkl'))
            
    def __process_indi_plan_csv_to_pickle(self,plan_df, wl_name, wl_group, cpu_num):
        mode = 0o666
        new_fname_prefix = '_'.join([wl_name, str(wl_group), str(cpu_num)])
        print(f"writing plan pkl to new file {new_fname_prefix}")

        new_dir = os.path.join(self.__pkl_fdn, new_fname_prefix)
        try:
            os.mkdir(new_dir, mode)
        except:
            pass
        plan_df.to_pickle(os.path.join(new_dir, 'query_plan.pkl'))


@update_class()
class ExprData2():
    def load_pickle(self, exclude_cpu=[]):
        '''
        lists of:
        1. workload number (1-9)
        2. workload type (tpcc, tpch, twitter)
        3. number of cpu
        4. dataframe from query_event_log.csv
        5. dataframe from query_info_log.csv
        '''
        self.wl_groups, self.wl_names, self.cpu_nums, self.query_event_dfs, self.query_perf_dfs, self.query_plan_dfs = [], [], [], [], [], []
        for subfd in os.listdir(self.__pkl_fdn):
            curr_subfd = os.path.join(self.__pkl_fdn, subfd)

            if os.path.isfile(curr_subfd):
                continue

            vals = subfd.split('_')
            if vals[2] in exclude_cpu:
                continue
            self.wl_names.append(vals[0])
            self.wl_groups.append(vals[1])
            # self.cpu_nums.append(int(vals[2]))
            self.cpu_nums.append(vals[2])

            for subf in os.listdir(curr_subfd):
                curr_f = os.path.join(curr_subfd, subf)
                if 'query_event.pkl' in subf:
                    df = pd.read_pickle(curr_f)
                    self.query_event_dfs.append(df)
                if 'query_plan.pkl' in subf:
                    df = pd.read_pickle(curr_f)
                    self.query_plan_dfs.append(df)
                if 'query_perf.pkl' in subf:
                    df = pd.read_pickle(curr_f)
                    self.query_perf_dfs.append(df)
            if len(self.query_event_dfs) < len(self.wl_names):
                self.query_event_dfs.append(None)
            if len(self.query_perf_dfs) < len(self.wl_names):
                self.query_perf_dfs.append(None)
        
        self.plan_feature_cols = self.query_plan_dfs[0].columns.to_list()

        # self.perf_feature_cols = self.query_perf_dfs[0].columns.to_list()
        
        self.plan_mtxs = [df[self.plan_feature_cols].to_numpy() for df in self.query_plan_dfs]
        self.perf_mtxs = [df[self.perf_feature_cols].to_numpy() if df is not None else None for df in self.query_perf_dfs]
        self.feature_cols = np.concatenate((self.plan_feature_cols, self.perf_feature_cols), axis=None)


# In[10]:


@update_class()
class ExprData2():
    def __manual_read_plan_csv(self, filename, seperator):
        with open(filename) as f:
            header = True
            prev_row = None
            for line in f:
                # manually seperate
                row = line.strip().split(seperator)
                if len(row) < 4:
                    if prev_row is None:
                        prev_row = row
                        continue
                    prev_row[-1] += row[0]
                    if len(row) > 1:
                        prev_row.append(row[1])
                        df.loc[len(df.index)] = prev_row
                        prev_row = None
                elif header:
                    df = pd.DataFrame(columns=[row])
                    header = False
                elif len(row) == 4:
                    df.loc[len(df.index)] = row
                elif len(row) > 4:
                    temp_plan = row[1:-2]
                    plan_string = seperator.join(temp_plan)
                    revised_row = [row[0], plan_string, row[-2], row[-1]] 
                    df.loc[len(df.index)] = revised_row
        return df


# In[11]:


@update_class()
class ExprData2():
    def __calc_lock(self, df_perf):
        start_time = 0
        
        lock_counts = []
        prev_lock = 0
        prev_time = 0

        lock_waits = []
        prev_lock_time = 0
        prev_lock_base = 0

        prev_row = None
        for index, perf_row in df_perf.iterrows():
            curr_time = perf_row['MS_TICKS']
            if start_time == 0 or curr_time == start_time: # First row, or row happens before the previous row, start again
                # lock_waits.append(0)
                lock_counts.append(0)
                start_time = curr_time
            elif curr_time == prev_time:
                lock_counts.append(0)
                # lock_waits.append(0)
            elif curr_time > prev_time:
                # happens after the previous row, calculate
                lock_counts.append(1.0*(perf_row['LOCK_REQUESTS']-prev_lock)/(perf_row['MS_TICKS']-prev_time))
                # lock_waits.append(1.0*(perf_row['LOCKS_AVG_WAIT_TIME']-prev_lock_time)/(perf_row['LOCKS_AVG_WAIT_TIME_BASE']-prev_lock_base))
            else:
                pass
            prev_lock=perf_row['LOCK_REQUESTS']
            prev_time=curr_time
            prev_lock_time=perf_row['LOCKS_AVG_WAIT_TIME']
            prev_lock_base=perf_row['LOCKS_AVG_WAIT_TIME_BASE']
            lock_waits.append(perf_row['LOCKS_AVG_WAIT_TIME']-perf_row['LOCKS_AVG_WAIT_TIME_BASE'])

        df_perf['LOCK_REQ_ABS'] = lock_counts
        df_perf['LOCK_WAIT_ABS'] = lock_waits
        
        return df_perf
        
    def process_csv_and_load(self, wl_group=0):
        self.wl_groups, self.wl_names, self.cpu_nums, self.query_event_dfs, self.query_plan_dfs, self.query_perf_dfs, self.plan_mtxs = [], [], [], [], [], [], []
        self.plan_feature_cols = None

        for subfd in os.listdir(self.__fdn):
            # if subfd not in ['tpcc', 'tpch', 'twitter']:
            if subfd not in ['tpcds']:
                continue
            curr_wl_name = subfd
            curr_path = os.path.join(self.__fdn, subfd)
            print(f"Workload: {curr_wl_name} ")

            # wl_group = int(subfd[len(self.__wl_group_prefix):])
            for subfd in os.listdir(curr_path):
                if 'run_' not in subfd:
                    continue
                run_idx = int(subfd.split('_')[1])
                print(f"Run: {run_idx}")
                
                curr_run = os.path.join(curr_path, subfd)
                cpu_num = 80
                wl_group += 1

                self.query_event_dfs.append(None)
                self.query_perf_dfs.append(None)
                self.wl_groups.append(wl_group)
                self.cpu_nums.append(cpu_num)
                self.wl_names.append(curr_wl_name)
                
                fv_df = pd.DataFrame()
                for subfd in os.listdir(curr_run):   
                    if 'xml_output_' in subfd:
                        curr_file = os.path.join(curr_run, subfd)
                        # too large to hold all plans
                        fv_df = self.__stream_plan_to_mtx(curr_file, fv_df)
                self.query_plan_dfs.append(fv_df)
                self.plan_mtxs.append(fv_df[self.plan_feature_cols].to_numpy())
                self.__process_indi_plan_csv_to_pickle(fv_df, curr_wl_name, wl_group, cpu_num)
                
        self.perf_mtxs = [df[self.perf_feature_cols].to_numpy() if df is not None else None for df in self.query_perf_dfs]
        self.feature_cols = np.concatenate((self.plan_feature_cols, self.perf_feature_cols), axis=None)

        self.__process_csv_to_pickle()

# In[12]:


@update_class()
class ExprData2():
    def split_by_sku(self):
        result = {cpu : ExprData2(wl_groups=[], wl_names=[], cpu_nums=[],
                                 query_event_dfs=[], query_plan_dfs=[], query_perf_dfs=[],
                                 plan_mtxs=[], perf_mtxs=[], 
                                 plan_feature_cols=self.plan_feature_cols, perf_feature_cols=self.perf_feature_cols) 
                  for cpu in np.unique(self.cpu_nums)}
        for i in range(len(self.wl_groups)):
            cpu = self.cpu_nums[i]
            result[cpu].wl_groups.append(self.wl_groups[i])
            result[cpu].wl_names.append(self.wl_names[i])
            result[cpu].cpu_nums.append(self.cpu_nums[i])
            result[cpu].query_event_dfs.append(self.query_event_dfs[i])
            result[cpu].query_plan_dfs.append(self.query_plan_dfs[i])
            result[cpu].query_perf_dfs.append(self.query_perf_dfs[i])
            result[cpu].plan_mtxs.append(self.plan_mtxs[i])
            result[cpu].perf_mtxs.append(self.perf_mtxs[i])
        return result
    
    def add_exprs(self, expr_b):
        self.cpu_nums += expr_b.cpu_nums
        self.wl_groups += expr_b.wl_groups
        self.wl_names += expr_b.wl_names
        self.query_event_dfs += expr_b.query_event_dfs
        self.query_plan_dfs += expr_b.query_event_dfs
        self.plan_mtxs += expr_b.plan_mtxs
        self.perf_mtxs += expr_b.perf_mtxs
    
    def keep_complete_exprs(self):
        result = ExprData2(wl_groups=[], wl_names=[], cpu_nums=[],
                         query_event_dfs=[], query_plan_dfs=[], query_perf_dfs=[],
                         plan_mtxs=[], perf_mtxs=[], 
                         plan_feature_cols=self.plan_feature_cols, perf_feature_cols=self.perf_feature_cols) 
        for i in range(len(self.wl_groups)):
            if self.query_event_dfs[i] is not None and self.query_plan_dfs[i] is not None and self.query_perf_dfs[i] is not None:
                result.wl_groups.append(self.wl_groups[i])
                result.wl_names.append(self.wl_names[i])
                result.cpu_nums.append(self.cpu_nums[i])
                result.query_event_dfs.append(self.query_event_dfs[i])
                result.query_plan_dfs.append(self.query_plan_dfs[i])
                result.query_perf_dfs.append(self.query_perf_dfs[i])
                result.plan_mtxs.append(self.plan_mtxs[i])
                result.perf_mtxs.append(self.perf_mtxs[i])
        return result

