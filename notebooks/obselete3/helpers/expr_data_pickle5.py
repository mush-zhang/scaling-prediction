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
import json
# import pickle
import pickle5 as pickle
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


class ExprDataPickle5():
    def __init__(self, 
                 wl_groups=None, wl_names=None, cpu_nums=None, run_idx=None, sampled_run_idx=None,
                 wl_throughput=None, wl_latency=None, terminal_num=None,
                 wl_throughput_samples=None, wl_latency_samples=None,
                 query_event_dfs=None, query_plan_dfs=None, query_perf_dfs=None,
                 plan_mtxs=None, perf_mtxs=None, plan_feature_cols=None, perf_feature_cols=None):
        self.__pkl_fdn = '../model/processed_wl_older_python/'
        self.__fdn = '../model/workloads/'
        self.__wl_group_prefix = 'workload_'
        self.__config_prefix = 'cpu'
        self.__names = ['tpcc', 'tpch', 'twitter', 'twitter', 'tpch', 'tpch', 'tpcc', 'tpcc', 'twitter', 'ycsb', 'ycsb', 'ycsb']
        self.wl_groups = wl_groups
        self.wl_names = wl_names
        self.cpu_nums = cpu_nums
        self.run_idx = run_idx
        self.sampled_run_idx = sampled_run_idx
        self.query_event_dfs = query_event_dfs
        self.query_perf_dfs = query_perf_dfs
        self.query_plan_dfs = query_plan_dfs
        self.plan_mtxs = plan_mtxs
        self.perf_mtxs = perf_mtxs
        self.wl_throughput = wl_throughput
        self.wl_latency = wl_latency
        self.terminal_num = terminal_num
        self.wl_throughput_samples = wl_throughput_samples
        self.wl_latency_samples = wl_latency_samples
        
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
class ExprDataPickle5():
    def get_num_exprs(self):
        return 0 if self.wl_groups is None else len(self.wl_groups)


# ### Functions to Parse Query Plans to Matrices
# 
# Helper functions to process query plan CSV files and extract the numerical values in plans to dataframes

# In[5]:


@update_class()
class ExprDataPickle5():
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
class ExprDataPickle5():
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
                if type(curr_p) != str:
                    curr_p = curr_p.values[0]
                if isinstance(curr_p, pd.Series):
                    curr_p = curr_p[0]
                plan_tree, plan_string = self.__parse_query_to_tree(curr_p)
                curr_dict = self.__extract_fv_from_tree(plan_tree)
                curr_df = pd.DataFrame([curr_dict])
                fv_df = pd.concat([fv_df, curr_df], axis=0, ignore_index=True)
                if not fea_col_filled:
                    self.plan_feature_cols = fv_df.columns
                    fea_col_filled = True
            mtxs.append(fv_df)

        self.query_plan_dfs = mtxs


# In[7]:


@update_class()
class ExprDataPickle5():
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

        iterable_all = zip(self.wl_groups, self.wl_names, self.cpu_nums, self.run_idx, self.query_event_dfs, self.query_perf_dfs, self.query_plan_dfs, self.wl_throughput, self.wl_latency, self.terminal_num, self.wl_throughput_samples, self.wl_latency_samples)
        for expr_num, wl_type, cpu_num, run_idx, qe_df, perf_df, plan_df, curr_thr, curr_lat, curr_term, curr_thr_samples, curr_lat_samples in iterable_all:
            new_fname_prefix = '_'.join([wl_type, str(expr_num), str(cpu_num), str(run_idx)])
            new_dir = os.path.join(self.__pkl_fdn, new_fname_prefix)
            try:
                os.mkdir(new_dir, mode)
            except:
                pass
            if qe_df is not None:
                qe_df.to_pickle(os.path.join(new_dir, 'query_event.pkl'))
            if perf_df is not None:
                perf_df.to_pickle(os.path.join(new_dir, 'query_perf.pkl'))
            plan_df.to_pickle(os.path.join(new_dir, 'query_plan.pkl'))
            with open(os.path.join(new_dir, 'throughput_samples.npy'), 'wb') as handle:
                np.save(handle, curr_thr_samples)
                # pickle.dump(curr_thr_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(new_dir, 'latency_samples.npy'), 'wb') as handle:
                np.save(handle, curr_thr_samples)
                # pickle.dump(curr_lat_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
            summary_dict = {
                'latency': [curr_lat],
                'throughput': [curr_thr],
                'terminal': [curr_term],
                # 'run_idx': [run_idx],
            }
            sum_df = pd.DataFrame.from_dict(summary_dict)
            sum_df.to_pickle(os.path.join(new_dir, 'summary.pkl'))
            # with open(os.path.join(new_dir, 'summary.pkl'), 'wb') as handle:
            #     pickle.dump(summary_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
# ### Functions to Parse Performance Info to Matrices

# In[8]:


@update_class()
class ExprDataPickle5():
    '''
    not used
    '''
    def __manual_read_perf_csv(self, fdn):
        # get throughput
        curr_th = []
        curr_lat = []
        curr_timed_throughputs=[]
        curr_timed_latencies=[]
        for subf in os.listdir(fdn):
            if 'summary.json' in subf:
                js_file = open(os.path.join(fdn, subf))
                js_data = json.load(js_file)

                curr_th.append(float(js_data["Throughput (requests/second)"]))
                curr_lat.append(float(js_data["Latency Distribution"]["Average Latency (microseconds)"])/1000)

            if 'results.csv' in subf:
                df_results = pd.read_csv(os.path.join(fdn, subf), header=0)
                curr_timed_throughputs += df_results['Throughput (requests/second)'].tolist()
                curr_timed_latencies += df_results['Average Latency (millisecond)'].tolist()
            curr_monitor = os.path.join(fdn, 'monitor')
            for subf in os.listdir(curr_monitor):
                # read queries
                if 'perf_event_log.csv' in subf:
                    # read perf
                    df_perf = pd.read_csv(os.path.join(curr_monitor, subf), header=0, parse_dates=True)
                    df_perf['TIMESTAMP'] = pd.to_datetime(df_perf['TIMESTAMP'],  format='%Y.%m.%d-%H:%M:%S:%f')

                    # perf cpu util
                    df_perf['CPU_UTILIZATION'] = (df_perf['CPU_USAGE_PERC'] / df_perf['CPU_USAGE_PERC_BASE']) * 100
                    df_perf['CPU_EFFECTIVE'] = (df_perf['CPU_EFFECTIVE_PERC'] / df_perf['CPU_EFFECTIVE_PERC_BASE']) * 100
                    df_perf['IOPS_TOTAL'] = df_perf['DISK_READ_IOPS'] + df_perf['DISK_WRITE_IOPS']
                    df_perf['READ_WRITE_RATIO'] = (df_perf['DISK_READ_IOPS'] / df_perf['DISK_WRITE_IOPS']) * 100
                    df_perf['MEM_UTILIZATION'] = (df_perf['USED_MEMORY'] / df_perf['TARGET_MEMORY']) * 100

                    df_perf = df_perf.sort_values(by='TIMESTAMP').drop_duplicates()

            # summary_df.loc[len(summary_df.index)] = [cpu_num, df_perf['CPU_UTILIZATION'].tolist(), df_perf['MEM_UTILIZATION'].to_list(), df_perf['IOPS_TOTAL'].tolist(), 
            #                                          sum(curr_th)/len(curr_th), curr_timed_throughputs, sum(curr_lat)/len(curr_lat), curr_timed_latencies, 
            #                                          df_perf['LOCK_REQ_ABS'][1:-1].tolist(), df_perf['LOCK_WAIT_ABS'][1:].tolist()]


# ### Load data
# ##### Public Functions

# In[9]:


@update_class()
class ExprDataPickle5():
    def load_pickle(self):
        '''
        lists of:
        1. workload number (1-9)
        2. workload type (tpcc, tpch, twitter)
        3. number of cpu
        4. dataframe from query_event_log.csv
        5. dataframe from query_info_log.csv
        '''
        self.wl_groups, self.wl_names, self.cpu_nums, self.terminal_num, self.run_idx, self.sampled_run_idx =  [], [], [], [], [], []
        self.query_event_dfs, self.query_perf_dfs, self.query_plan_dfs = [], [], []
        self.wl_throughput, self.wl_latency = [], []
        self.wl_throughput_samples, self.wl_latency_samples =  [], []
        
        for subfd in os.listdir(self.__pkl_fdn):
            curr_subfd = os.path.join(self.__pkl_fdn, subfd)

            if os.path.isfile(curr_subfd) or 'hyperscale' in subfd:
                continue

            vals = subfd.split('_')
            self.wl_names.append(vals[0])
            self.wl_groups.append(vals[1])
            self.cpu_nums.append(vals[2])
            self.run_idx.append(vals[3])
            self.sampled_run_idx.append(vals[3])

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
                if 'summary.pkl' in subf:
                    df = pd.read_pickle(curr_f)
                    self.wl_throughput.append(df['throughput'].to_list()[0])
                    self.wl_latency.append(df['latency'].to_list()[0])
                    self.terminal_num.append(df['terminal'].to_list()[0])
                if 'throughput_samples.npy' in subf:
                    self.wl_throughput_samples.append(np.load(curr_f, allow_pickle=True))
                if 'latency_samples.npy' in subf:
                    self.wl_latency_samples.append(np.load(curr_f, allow_pickle=True))
            if len(self.query_event_dfs) < len(self.wl_names):
                self.query_event_dfs.append(None)
            if len(self.query_perf_dfs) < len(self.wl_names):
                self.query_perf_dfs.append(None)
            if len(self.wl_throughput) < len(self.wl_names):
                self.wl_throughput.append(None)
            if len(self.wl_latency) < len(self.wl_names):
                self.wl_latency.append(None)
            if len(self.terminal_num) < len(self.wl_names):
                self.terminal_num.append(None)
            if len(self.run_idx) < len(self.run_idx):
                self.run_idx.append(None)
        self.plan_feature_cols = self.query_plan_dfs[0].columns.to_list()
        
        self.plan_mtxs = [df[self.plan_feature_cols].to_numpy() for df in self.query_plan_dfs]
        self.perf_mtxs = [df[self.perf_feature_cols].to_numpy() if df is not None else None for df in self.query_perf_dfs]
        self.feature_cols = np.concatenate((self.plan_feature_cols, self.perf_feature_cols), axis=None)


# In[10]:


@update_class()
class ExprDataPickle5():
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
                    df = pd.DataFrame(columns=[row], dtype='str')
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
class ExprDataPickle5():
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
        
    def process_csv_and_load(self):
        self.wl_groups, self.wl_names, self.cpu_nums, self.run_idx, self.sampled_run_idx = [], [], [], [], []
        self.query_event_dfs, self.query_plan_dfs, self.query_perf_dfs = [], [], []
        self.wl_throughput, self.wl_latency, self.terminal_num, self.wl_throughput_samples, self.wl_latency_samples = [], [], [], [], []
        
        run_idx = 0
        
        for subfd in os.listdir(self.__fdn):
            curr_wl = os.path.join(self.__fdn, subfd)

            if os.path.isfile(curr_wl) or self.__wl_group_prefix not in subfd:
                continue
            wl_group = int(subfd[len(self.__wl_group_prefix):])
            for subfd in os.listdir(curr_wl):
                if '_remote_' not in subfd:
                    continue
                curr_run = os.path.join(curr_wl, subfd)
                curr_run_idx = run_idx
                run_idx = run_idx + 1
                curr_wl_name = self.__names[wl_group-1]
                print(curr_run_idx, curr_run, curr_wl_name)
                                
                # if curr_wl_name != 'ycsb':
                #     continue
                for subfd in os.listdir(curr_run):
                    curr_config = os.path.join(curr_run, subfd)
                    if os.path.isfile(curr_config):
                        continue
                    # cpu_num = int(subfd[len(self.__config_prefix):])
                    cpu_num = 'cpu{}'.format(subfd[len(self.__config_prefix):])

                    # curr_wl_name = self.__names[wl_group-1]
                    curr_monitor = os.path.join(curr_config, 'w_monitor/results/monitor/')
                    curr_result = os.path.join(curr_config, 'w_monitor/results/')

                    # get throughput
                    curr_thr = None
                    curr_lat = None
                    curr_term = None
                    df_result = None
                    for subf in os.listdir(curr_result):
                        if 'summary.json' in subf:
                            js_file = open(os.path.join(curr_result, subf))
                            js_data = json.load(js_file)
            
                            curr_thr = float(js_data["Throughput (requests/second)"])
                            curr_lat = float(js_data["Latency Distribution"]["Average Latency (microseconds)"])/1000
                        if 'config.xml' in subf:
                            tree = ET.parse(os.path.join(curr_result, subf))
                            root = tree.getroot()
                            for child in root:
                                if child.tag == 'terminals':
                                    curr_term = int(child.text)
                                    break
                        if 'samples.csv' in subf:
                            df_result = pd.read_csv(os.path.join(curr_result, subf), sep=',', header=0)
                            df_result = df_result[['Throughput (requests/second)', 'Average Latency (microseconds)']]
                            df_result['Average Latency (microseconds)'] = df_result['Average Latency (microseconds)']/1000.0
                    self.wl_throughput.append(curr_thr)
                    self.wl_latency.append(curr_lat)
                    self.terminal_num.append(curr_term)
                    # print(curr_term)
                    self.wl_throughput_samples.append(df_result['Throughput (requests/second)'].to_list())
                    self.wl_latency_samples.append(df_result['Average Latency (microseconds)'].to_list())
                    
                    for subfd in os.listdir(curr_monitor):
                        if 'query_event_log.csv' in subfd:
                            curr_file = os.path.join(curr_monitor, subfd)
                            df = pd.read_csv(curr_file, sep=',', header=0)
                            first_column = df.pop('TIMESTAMP')
                            df.drop([ 'PLAN_HANDLE', 'IDENTIFIER'], axis=1, inplace=True)
                            df.insert(0, 'TIMESTAMP', first_column)
                            self.query_event_dfs.append(df)
                            # query_event_dfs.append(df.loc[:, df.columns != 'TIMESTAMP'].to_numpy())
                            self.wl_groups.append(wl_group)
                            self.run_idx.append(curr_run_idx)         
                            self.sampled_run_idx.append(run_idx)
                            self.cpu_nums.append(cpu_num)
                            self.wl_names.append(curr_wl_name)
                        if 'query_info_log.csv' in subfd:
                            curr_file = os.path.join(curr_monitor, subfd)
                            if curr_wl_name == 'tpch' or curr_wl_name == 'ycsb':
                                sep = '||'
                            else:
                                sep = '|'
                            df2 = self.__manual_read_plan_csv(curr_file, sep)
                            if (df2.shape[1] != 4):
                                print("this should not happen")
                            self.query_plan_dfs.append(df2)
                        if 'perf_event_log.csv' in subfd:
                            # read perf
                            df_perf = pd.read_csv(os.path.join(curr_monitor, subfd), header=0, parse_dates=True)
                            df_perf['TIMESTAMP'] = pd.to_datetime(df_perf['TIMESTAMP'],  format='%Y.%m.%d-%H:%M:%S:%f')

                            # perf cpu util
                            df_perf['CPU_UTILIZATION'] = (df_perf['CPU_USAGE_PERC'] / df_perf['CPU_USAGE_PERC_BASE']) * 100
                            df_perf['CPU_EFFECTIVE'] = (df_perf['CPU_EFFECTIVE_PERC'] / df_perf['CPU_EFFECTIVE_PERC_BASE']) * 100
                            df_perf['IOPS_TOTAL'] = df_perf['DISK_READ_IOPS'] + df_perf['DISK_WRITE_IOPS']
                            df_perf['READ_WRITE_RATIO'] = (df_perf['DISK_READ_IOPS'] / df_perf['DISK_WRITE_IOPS']) * 100
                            df_perf['MEM_UTILIZATION'] = (df_perf['USED_MEMORY'] / df_perf['TARGET_MEMORY']) * 100
                            df_perf = self.__calc_lock(df_perf)

                            df_perf = df_perf.sort_values(by='TIMESTAMP').drop_duplicates()
                            df_perf.replace([np.inf, -np.inf], 0, inplace=True)
                            df_perf.fillna(0, inplace=True)
                            df_perf = df_perf[self.perf_feature_cols]
                            self.query_perf_dfs.append(df_perf)

        fdn='../model/new_dataset'
        wl_group_prefix = 'xml_ter_'
        config_prefix = 'xml_ter_'
        idx = len(np.unique(self.wl_groups))+1
        for subfd in os.listdir(fdn):
            # xml_ter_?
            curr_wl = os.path.join(fdn, subfd)

            if os.path.isfile(curr_wl) or wl_group_prefix not in subfd:
                continue

            wl_group = idx
            idx = idx + 1
            terminal_num = 'ter{}'.format(subfd[len(config_prefix):])
            curr_wl_name = 'xml'

            self.query_event_dfs.append(None)
            self.query_perf_dfs.append(None)
            self.wl_groups.append(wl_group)
            self.cpu_nums.append(terminal_num)
            self.run_idx.append(run_idx)
            self.sampled_run_idx.append(run_idx)
            run_idx = run_idx + 1
            self.wl_names.append(curr_wl_name)
            self.wl_throughput.append(None)
            self.wl_latency.append(None)
            self.terminal_num.append(None)
            self.wl_throughput_samples.append(None)
            self.wl_latency_samples.append(None)
            plans = []
            for subfd in os.listdir(curr_wl):
                # xml_output_?.txt
                curr_config = os.path.join(curr_wl, subfd)

                if 'xml_output_' in subfd:
                    curr_file = os.path.join(curr_wl, subfd)
                    with open(curr_file) as f:
                        curr_plans = f.readlines()
                        plans = plans + curr_plans
            self.query_plan_dfs.append(pd.DataFrame(plans, columns = ['QUERY_PLAN']))
        self.__plan_to_mtx()

        self.plan_mtxs = [df[self.plan_feature_cols].to_numpy() for df in self.query_plan_dfs]
        self.perf_mtxs = [df[self.perf_feature_cols].to_numpy() if df is not None else None for df in self.query_perf_dfs]
        self.feature_cols = np.concatenate((self.plan_feature_cols, self.perf_feature_cols), axis=None)

        self.__process_csv_to_pickle()

# In[12]:


@update_class()
class ExprDataPickle5():
    def split_by_sku(self):
        result = {cpu : ExprDataPickle5(wl_groups=[], wl_names=[], cpu_nums=[], run_idx=[], sampled_run_idx = [],
                                 wl_throughput=[], wl_latency=[], terminal_num=[],
                                 wl_throughput_samples=[], wl_latency_samples=[],
                                 query_event_dfs=[], query_plan_dfs=[], query_perf_dfs=[],
                                 plan_mtxs=[], perf_mtxs=[], 
                                 plan_feature_cols=self.plan_feature_cols, perf_feature_cols=self.perf_feature_cols) 
                  for cpu in np.unique(self.cpu_nums)}
        for i in range(len(self.wl_groups)):
            cpu = self.cpu_nums[i]
            result[cpu].wl_groups.append(self.wl_groups[i])
            result[cpu].wl_names.append(self.wl_names[i])
            result[cpu].cpu_nums.append(self.cpu_nums[i])
            result[cpu].run_idx.append(self.run_idx[i])
            result[cpu].sampled_run_idx.append(self.sampled_run_idx[i])
            result[cpu].query_event_dfs.append(self.query_event_dfs[i])
            result[cpu].query_plan_dfs.append(self.query_plan_dfs[i])
            result[cpu].query_perf_dfs.append(self.query_perf_dfs[i])
            result[cpu].plan_mtxs.append(self.plan_mtxs[i])
            result[cpu].perf_mtxs.append(self.perf_mtxs[i])
            result[cpu].wl_throughput.append(self.wl_throughput[i])
            result[cpu].wl_latency.append(self.wl_latency[i])
            result[cpu].terminal_num.append(self.terminal_num[i])
            result[cpu].wl_throughput_samples.append(self.wl_throughput_samples[i])
            result[cpu].wl_latency_samples.append(self.wl_latency_samples[i])
        return result

    def split_by_type(self):
        result = {expr : ExprDataPickle5(wl_groups=[], wl_names=[], cpu_nums=[], run_idx=[], sampled_run_idx = [],
                                 wl_throughput=[], wl_latency=[], terminal_num=[],
                                 wl_throughput_samples=[], wl_latency_samples=[],
                                 query_event_dfs=[], query_plan_dfs=[], query_perf_dfs=[],
                                 plan_mtxs=[], perf_mtxs=[], 
                                 plan_feature_cols=self.plan_feature_cols, perf_feature_cols=self.perf_feature_cols) 
                  for expr in np.unique(self.wl_groups)}
        for i in range(len(self.wl_groups)):
            group = self.wl_groups[i]
            result[group].wl_groups.append(self.wl_groups[i])
            result[group].wl_names.append(self.wl_names[i])
            result[group].cpu_nums.append(self.cpu_nums[i])
            result[group].run_idx.append(self.run_idx[i])
            result[group].sampled_run_idx.append(self.sampled_run_idx[i])
            result[group].query_event_dfs.append(self.query_event_dfs[i])
            result[group].query_plan_dfs.append(self.query_plan_dfs[i])
            result[group].query_perf_dfs.append(self.query_perf_dfs[i])
            result[group].plan_mtxs.append(self.plan_mtxs[i])
            result[group].perf_mtxs.append(self.perf_mtxs[i])
            result[group].wl_throughput.append(self.wl_throughput[i])
            result[group].wl_latency.append(self.wl_latency[i])
            result[group].terminal_num.append(self.terminal_num[i])
            result[group].wl_throughput_samples.append(self.wl_throughput_samples[i])
            result[group].wl_latency_samples.append(self.wl_latency_samples[i])
        return result
    
    def split_by_expr(self):
        result = {expr : ExprDataPickle5(wl_groups=[], wl_names=[], cpu_nums=[], run_idx=[], sampled_run_idx = [],
                                 wl_throughput=[], wl_latency=[], terminal_num=[],
                                 wl_throughput_samples=[], wl_latency_samples=[],
                                 query_event_dfs=[], query_plan_dfs=[], query_perf_dfs=[],
                                 plan_mtxs=[], perf_mtxs=[], 
                                 plan_feature_cols=self.plan_feature_cols, perf_feature_cols=self.perf_feature_cols) 
                  for expr in np.unique(self.run_idx)}
        for i in range(len(self.run_idx)):
            group = self.run_idx[i]
            result[group].wl_groups.append(self.wl_groups[i])
            result[group].wl_names.append(self.wl_names[i])
            result[group].cpu_nums.append(self.cpu_nums[i])
            result[group].run_idx.append(self.run_idx[i])
            result[group].sampled_run_idx.append(self.sampled_run_idx[i])
            result[group].query_event_dfs.append(self.query_event_dfs[i])
            result[group].query_plan_dfs.append(self.query_plan_dfs[i])
            result[group].query_perf_dfs.append(self.query_perf_dfs[i])
            result[group].plan_mtxs.append(self.plan_mtxs[i])
            result[group].perf_mtxs.append(self.perf_mtxs[i])
            result[group].wl_throughput.append(self.wl_throughput[i])
            result[group].wl_latency.append(self.wl_latency[i])
            result[group].terminal_num.append(self.terminal_num[i])
            result[group].wl_throughput_samples.append(self.wl_throughput_samples[i])
            result[group].wl_latency_samples.append(self.wl_latency_samples[i])
        return result
    
    def split_by_term(self):
        result = {expr : ExprDataPickle5(wl_groups=[], wl_names=[], cpu_nums=[], run_idx=[], sampled_run_idx = [],
                                 wl_throughput=[], wl_latency=[], terminal_num=[],
                                 wl_throughput_samples=[], wl_latency_samples=[],
                                 query_event_dfs=[], query_plan_dfs=[], query_perf_dfs=[],
                                 plan_mtxs=[], perf_mtxs=[], 
                                 plan_feature_cols=self.plan_feature_cols, perf_feature_cols=self.perf_feature_cols) 
                  for expr in np.unique(self.terminal_num)}
        for i in range(len(self.terminal_num)):
            group = self.terminal_num[i]
            result[group].wl_groups.append(self.wl_groups[i])
            result[group].wl_names.append(self.wl_names[i])
            result[group].cpu_nums.append(self.cpu_nums[i])
            result[group].run_idx.append(self.run_idx[i])
            result[group].sampled_run_idx.append(self.sampled_run_idx[i])
            result[group].query_event_dfs.append(self.query_event_dfs[i])
            result[group].query_plan_dfs.append(self.query_plan_dfs[i])
            result[group].query_perf_dfs.append(self.query_perf_dfs[i])
            result[group].plan_mtxs.append(self.plan_mtxs[i])
            result[group].perf_mtxs.append(self.perf_mtxs[i])
            result[group].wl_throughput.append(self.wl_throughput[i])
            result[group].wl_latency.append(self.wl_latency[i])
            result[group].terminal_num.append(self.terminal_num[i])
            result[group].wl_throughput_samples.append(self.wl_throughput_samples[i])
            result[group].wl_latency_samples.append(self.wl_latency_samples[i])
        return result
    
    def add_exprs(self, expr_b):
        self.cpu_nums += expr_b.cpu_nums
        self.run_idx += expr_b.run_idx
        self.sampled_run_idx += expr_b.sampled_run_idx
        self.wl_groups += expr_b.wl_groups
        self.wl_names += expr_b.wl_names
        self.query_event_dfs += expr_b.query_event_dfs
        self.query_plan_dfs += expr_b.query_event_dfs
        self.plan_mtxs += expr_b.plan_mtxs
        self.perf_mtxs += expr_b.perf_mtxs
        self.wl_throughput += expr_b.wl_throughput
        self.wl_latency += expr_b.wl_latency
        self.terminal_num += expr_b.terminal_num
        self.wl_throughput_samples += expr_b.wl_throughput_samples
        self.wl_latency_samples += expr_b.wl_latency_samples   

    def __add_expr_from_self(self, result, i):
        result.wl_groups.append(self.wl_groups[i])
        result.wl_names.append(self.wl_names[i])
        result.cpu_nums.append(self.cpu_nums[i])
        result.run_idx.append(self.run_idx[i])
        result.sampled_run_idx.append(self.sampled_run_idx[i])
        result.query_event_dfs.append(self.query_event_dfs[i])
        result.query_plan_dfs.append(self.query_plan_dfs[i])
        result.query_perf_dfs.append(self.query_perf_dfs[i])
        result.plan_mtxs.append(self.plan_mtxs[i])
        result.perf_mtxs.append(self.perf_mtxs[i])
        result.wl_throughput.append(self.wl_throughput[i])
        result.wl_latency.append(self.wl_latency[i])
        result.terminal_num.append(self.terminal_num[i])
        result.wl_throughput_samples.append(self.wl_throughput_samples[i])
        result.wl_latency_samples.append(self.wl_latency_samples[i])
        return result
        
    def keep_complete_exprs(self):
        result = ExprDataPickle5(wl_groups=[], wl_names=[], cpu_nums=[], run_idx=[],  sampled_run_idx = [],
                         wl_throughput=[], wl_latency=[], terminal_num=[],
                         wl_throughput_samples=[], wl_latency_samples=[],
                         query_event_dfs=[], query_plan_dfs=[], query_perf_dfs=[],
                         plan_mtxs=[], perf_mtxs=[], 
                         plan_feature_cols=self.plan_feature_cols, perf_feature_cols=self.perf_feature_cols) 
        print(len(self.wl_groups), len(self.query_event_dfs), len(self.query_plan_dfs), len(self.query_perf_dfs))
        for i in range(len(self.wl_groups)):
            if self.query_event_dfs[i] is not None and self.query_plan_dfs[i] is not None and self.query_perf_dfs[i] is not None:
                result = self.__add_expr_from_self(result, i)
        return result

    def __calculate_num_samples(self, length):
        num_samples = length/10
        num_samples = np.max([num_samples, np.min([25, length/5])])
        num_samples = np.min([num_samples, 1000])
        return int(num_samples)

    def __sample_helper(self, length, num_samples):
        return np.random.choice(length, num_samples)

    def sample_data(self, num_sample_per_run=10):
        result = ExprDataPickle5(wl_groups=[], wl_names=[], cpu_nums=[], run_idx=[], sampled_run_idx = [],
                         wl_throughput=[], wl_latency=[], terminal_num=[],
                         wl_throughput_samples=[], wl_latency_samples=[],
                         query_event_dfs=[], query_plan_dfs=[], query_perf_dfs=[],
                         plan_mtxs=[], perf_mtxs=[], 
                         plan_feature_cols=self.plan_feature_cols, perf_feature_cols=self.perf_feature_cols) 
        sampled_run_idx = 0;
        for j in range(num_sample_per_run):

            for run in np.unique(self.run_idx):
                curr_list = [i for i, value in enumerate(self.run_idx) if value == run]
                for i in curr_list:
                    if self.wl_names[i] == 'xml':
                        # not sample
                        result = self.__add_expr_from_self(result, i)
                        continue
        
                    query_event_length = self.query_event_dfs[i].shape[0];
                    query_event_sample_size = self.__calculate_num_samples(query_event_length)
                    perf_length = self.query_perf_dfs[i].shape[0]
                    perf_sample_size = self.__calculate_num_samples(perf_length)
                    plan_length = self.query_plan_dfs[i].shape[0]
                    plan_sample_size = self.__calculate_num_samples(plan_length)
                
                    thr_lat_length = len(self.wl_throughput_samples[i])
                    thr_lat_sample_size = self.__calculate_num_samples(thr_lat_length)
    
                    result.wl_groups.append(self.wl_groups[i])
                    result.wl_names.append(self.wl_names[i])
                    result.cpu_nums.append(self.cpu_nums[i])
                    result.run_idx.append(self.run_idx[i])
                    result.sampled_run_idx.append(sampled_run_idx)
                    result.terminal_num.append(self.terminal_num[i])
                    
                    result.query_event_dfs.append(self.query_event_dfs[i].iloc[self.__sample_helper(query_event_length, query_event_sample_size)])
                    plan_idxs = self.__sample_helper(plan_length, plan_sample_size)
                    perf_idxs = self.__sample_helper(perf_length, perf_sample_size)
                    result.query_plan_dfs.append(self.query_plan_dfs[i].iloc[plan_idxs])
                    result.query_perf_dfs.append(self.query_perf_dfs[i].iloc[perf_idxs])
                    result.plan_mtxs.append(np.array(self.plan_mtxs[i])[plan_idxs.astype(int)])
                    result.perf_mtxs.append(np.array(self.perf_mtxs[i])[perf_idxs.astype(int)])
            
                    thr_lat_idxs = self.__sample_helper(thr_lat_length, thr_lat_sample_size)
                    wl_thr_samples = np.array(self.wl_throughput_samples[i])[thr_lat_idxs]
                    wl_lat_samples = np.array(self.wl_latency_samples[i])[thr_lat_idxs]
                    result.wl_throughput_samples.append(wl_thr_samples)
                    result.wl_latency_samples.append(wl_lat_samples)
                    result.wl_throughput.append(np.mean(wl_thr_samples))
                    result.wl_latency.append(np.mean(wl_lat_samples))
                sampled_run_idx += 1
        return result
    def remove_by_group(self, group_list_to_remove):
        result = ExprDataPickle5(wl_groups=[], wl_names=[], cpu_nums=[], run_idx=[], sampled_run_idx = [],
                         wl_throughput=[], wl_latency=[], terminal_num=[],
                         wl_throughput_samples=[], wl_latency_samples=[],
                         query_event_dfs=[], query_plan_dfs=[], query_perf_dfs=[],
                         plan_mtxs=[], perf_mtxs=[], 
                         plan_feature_cols=self.plan_feature_cols, perf_feature_cols=self.perf_feature_cols) 
        for i in range(len(self.wl_groups)):
            if self.wl_groups[i] in group_list_to_remove:
                continue
            result = self.__add_expr_from_self(result, i)
        return result
    
    def remove_by_wlname(self, wlnames_to_remove):
        result = ExprDataPickle5(wl_groups=[], wl_names=[], cpu_nums=[], run_idx=[], sampled_run_idx = [],
                         wl_throughput=[], wl_latency=[], terminal_num=[],
                         wl_throughput_samples=[], wl_latency_samples=[],
                         query_event_dfs=[], query_plan_dfs=[], query_perf_dfs=[],
                         plan_mtxs=[], perf_mtxs=[], 
                         plan_feature_cols=self.plan_feature_cols, perf_feature_cols=self.perf_feature_cols) 
        for i in range(len(self.wl_groups)):
            if self.wl_names[i] in wlnames_to_remove:
                continue
            result = self.__add_expr_from_self(result, i)
        return result

        
    def get_by_run_idx(self, run_idx):
        result = ExprDataPickle5(wl_groups=[], wl_names=[], cpu_nums=[], run_idx=[], sampled_run_idx = [],
                         wl_throughput=[], wl_latency=[], terminal_num=[],
                         wl_throughput_samples=[], wl_latency_samples=[],
                         query_event_dfs=[], query_plan_dfs=[], query_perf_dfs=[],
                         plan_mtxs=[], perf_mtxs=[], 
                         plan_feature_cols=self.plan_feature_cols, perf_feature_cols=self.perf_feature_cols) 
        for i in range(len(self.wl_groups)):
            if self.run_idx[i] == run_idx:
                result = self.__add_expr_from_self(result, i)
        return result
    
    def merge_tpch(self):
        result = ExprDataPickle5(wl_groups=[], wl_names=[], cpu_nums=[], run_idx=[], sampled_run_idx = [],
                     wl_throughput=[], wl_latency=[], terminal_num=[],
                     wl_throughput_samples=[], wl_latency_samples=[],
                     query_event_dfs=[], query_plan_dfs=[], query_perf_dfs=[],
                     plan_mtxs=[], perf_mtxs=[], 
                     plan_feature_cols=self.plan_feature_cols, perf_feature_cols=self.perf_feature_cols)
        tpch_wl_group = None
        for i in range(len(self.wl_groups)):
            result = self.__add_expr_from_self(result, i)
            if self.wl_names[i] == 'tpch':
                if tpch_wl_group is None:
                    tpch_wl_group = self.wl_groups[i]
                result.wl_groups[-1] = tpch_wl_group
                result.terminal_num[-1] = 1
        return result

    def fix_tpch(self):
        result = ExprDataPickle5(wl_groups=[], wl_names=[], cpu_nums=[], run_idx=[], sampled_run_idx = [],
                     wl_throughput=[], wl_latency=[], terminal_num=[],
                     wl_throughput_samples=[], wl_latency_samples=[],
                     query_event_dfs=[], query_plan_dfs=[], query_perf_dfs=[],
                     plan_mtxs=[], perf_mtxs=[], 
                     plan_feature_cols=self.plan_feature_cols, perf_feature_cols=self.perf_feature_cols)
        for i in range(len(self.wl_groups)):
            result = self.__add_expr_from_self(result, i)
            if self.wl_names[i] == 'tpch':
                result.terminal_num[-1] = 1
        return result