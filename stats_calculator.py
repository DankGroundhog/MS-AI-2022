import argparse
import json
import logging
from operator import index
import re
import sys, fnmatch
from attr import attributes
from matplotlib.pyplot import pause

import numpy as np
import pandas as pd
import json

import onnx
from onnx import ModelProto, helper, numpy_helper, onnx_pb
from sqlalchemy import column

import os

'''
Script to calculate the aggregate ratios of all operators
through the model folder. Returns a list of (Operator, Ratio).

(Op_type, ratio) - Summarized option (default)
(Op_name, ratio) - Verbose option (-v flag)
'''

def get_args():
    parser = argparse.ArgumentParser(description='onnxruntime bench tool')
    parser.add_argument('--input', required=True, help='csv output')
    parser.add_argument('-v', action='store_true', help='verbose option, defaults to summarized')
    args = parser.parse_args()
    return args

def calculator(dir, model_dir, flag):
    os.chdir(f'{model_dir}')
    keyset = set()
    op_vals = [] # (Op_type, ratio)
    if flag: # if verbose option is selected
        for model in os.listdir():
            os.chdir(f'{dir}/{model_dir}/{model}/model_traces') 
            for log in os.listdir():
                if fnmatch.fnmatch(log, "*_verbose.csv"):
                    with open(log, 'r'):
                        data = pd.read_csv(log, usecols=["name", "count", 'op_type', 'input_type_shape', 'output_type_shape', 'attribute'])
                        # log.close()
                    for value in data['name'].values: keyset.add(value)
                    for op in range(len(data)):
                        op_vals.append([data['name'][op], data['count'][op]])

        # Enter the ratio calculation part of the calculator
        total_sum = 0
        stats_dict = dict()
        keyset = list(keyset)
        
        for key in range(len(keyset)): stats_dict[f'{keyset[key]}'] = 0 

        for i in range(len(stats_dict)):
            for j in range(len(op_vals)):
                temp = op_vals[j]
                if list(stats_dict)[i] == temp[0]:
                    total_sum += temp[1]
            stats_dict[f'{keyset[i]}'] = total_sum
            total_sum = 0
        del total_sum

        op_ratios = []
        vals = list(stats_dict.values())
        absolute_sum = sum(stats_dict.values())
        for op in range(len(stats_dict)):
            op_ratios.append(round((vals[op] / absolute_sum)*100, 3))

        stats_df = pd.DataFrame(index=keyset)
        stats_df["Total Count"] = stats_dict.values()
        stats_df["Op Ratio"] = op_ratios
        stats_df.sort_values(by="Total Count")
        # print(stats_df)
        # return stats_df
        os.chdir("../..")

    else: # default version (summarized)
        for model in os.listdir():
            os.chdir(f"{dir}/{model_dir}/{model}/model_traces")
            for log in os.listdir():
                if fnmatch.fnmatch(log, "*_summarized.csv"):
                    with open(log, 'r'):
                        data = pd.read_csv(log, usecols=["op_type", "count"])
                    for value in data['op_type'].values: keyset.add(value)

                    for op in range(len(data)):
                        op_vals.append((data['op_type'][op], data['count'][op]))

        # Enter the ratio calculation part of the calculator
        total_sum = 0
        stats_dict = dict()
        keyset = list(keyset)
        
        for key in range(len(keyset)): stats_dict[f'{keyset[key]}'] = 0 

        for i in range(len(stats_dict)):
            for j in range(len(op_vals)):
                temp = op_vals[j]
                if list(stats_dict)[i] == temp[0]:
                    total_sum += temp[1]
            stats_dict[f'{keyset[i]}'] = total_sum
            total_sum = 0
        del total_sum

        op_ratios = []
        vals = list(stats_dict.values())
        absolute_sum = sum(stats_dict.values())
        for op in range(len(stats_dict)):
            op_ratios.append(round((vals[op] / absolute_sum)*100, 3))

        stats_df = pd.DataFrame(index=keyset)
        stats_df["Total Count"] = stats_dict.values()
        stats_df["Op Ratio"] = op_ratios
        stats_df.sort_values(by="Total Count")
        # print(stats_df)
        # return stats_df
        os.chdir("../..")


if __name__ == "__main__":
    args = get_args()
    calculator(args.input, args.v)