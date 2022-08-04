import argparse
import json
import logging
import re
import sys
from attr import attributes
from matplotlib.pyplot import pause

import numpy as np
import pandas as pd
import json

import onnx
from onnx import ModelProto, helper, numpy_helper, onnx_pb
from sqlalchemy import column

import os

from getAttribs import attribs, set_attrib_json

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)  # pylint: disable=invalid-name


def get_args():
    parser = argparse.ArgumentParser(description='onnxruntime bench tool')
    parser.add_argument('--input', required=True, help='chrome trace json')
    parser.add_argument('--name', help='filter list')
    parser.add_argument('--csv', help='save intermidiate data to csv', action='store_true')
    parser.add_argument('-l', type=int, default=20, help='list top N items, default=20')
    parser.add_argument('-v', action='store_true', help='verbose')
    parser.add_argument('--nodes', action='store_true', help='show top N nodes')
    parser.add_argument('--source', required=False)
    args = parser.parse_args()
    return args


def json_to_df(profile_path, verbose):
    entries = []
    all_cats = set()
    all_names = set()
    with open(profile_path, "r") as f:
        data = json.load(f)

    if type(data) == dict:
        data = data['traceEvents']
    for item in data:
        dur = item.get("dur")
        if not dur:
            continue
        cat = item.get("cat")
        if cat not in ["Node", "Op"]:
            continue
        arg = item.get('args')
        if not arg:
            continue
        provider = arg.get("provider")
        op = arg.get("op_name")
        if op:
            name = item['name']
            if not name.endswith("_kernel_time"):
                continue
            dur = item['dur']
            name = name.replace("_kernel_time", "")
            # graph_index = arg.get('graph_index')
            parameter_size = arg.get('parameter_size')
            activation_size = arg.get('activation_size')
            output_size = arg.get('output_size')
            input_type_shape = arg.get('input_type_shape')
            input_type_shape = json.dumps(input_type_shape) # Use JSON decode when reading this field
            assert input_type_shape is not None
            output_type_shape = arg.get('output_type_shape')
            output_type_shape = json.dumps(output_type_shape) # Use JSON decode when reading this field
            assert input_type_shape is not None

            e = {
                "name": name, "dur": dur, "op_type": op, "provider": provider,
                "parameter_size": parameter_size, "activation_size": activation_size,
                "output_size": output_size, "input_type_shape": input_type_shape,
                "output_type_shape": output_type_shape
            }
            entries.append(e)
    df = pd.DataFrame([f for f in entries])
    df['count'] = 1
    return df


def logger(args):
    df = json_to_df(args.input, args.v)

    with open(f"{args.source}", "rb") as f:
        data = f.read()
        model_proto = ModelProto()
        model_proto.ParseFromString(data)
    
    df = pd.DataFrame(df) 

    digits = 1
    top = args.l
    pd.set_option('display.max_colwidth', 120)
    df2 = df[['dur', 'count']].sum()
    df['pct'] = (100 * df['dur'] / df2['dur'])

    if not args.nodes:
        fields = ["name", "op_type", "dur", "count", "pct", "input_type_shape", "output_type_shape"]
        
        # Summarized CSV
        df1 = df[fields].groupby(['op_type']).sum()
        df_attribs = attribs(model_proto)

        if (set(df.name) - set(df_attribs.name)) == set():
            print("No naming mismatches, proceeding...")
        else:
            print("There are naming conflicts, results may be slightly different from the model file. Proceed with caution...")
        df2 = pd.merge(df, df_attribs, on="name", how='outer')


        df1 = df1.sort_values(by="dur", ascending=False)
        df2 = df2.sort_values(by="I/O", ascending=False)
        df1['csum'] = df1['pct'].cumsum()
        df1['avg'] = df1['dur'] / df1['count']
        df2['csum'] = df2['pct'].cumsum()
        df2['avg'] = df2['dur'] / df2['count']

        # print("\n--Check model directories for CSV/JSON outputs--\n--Name:model_traces--\n")
        # print(df1)

    else:
        fields = ["name", "op_type", "dur", "count", "pct", "input_type_shape", "output_type_shape"]
        
        # Summarized CSV
        df1 = df[fields].groupby(['op_type']).sum()
        # Verbose CSV
        df_attribs = attribs(model_proto)
        if (set(df.name) - set(df_attribs.name)) == set():
            print("No naming mismatches, proceeding...")
        else:
            print("There are naming conflicts, results may be slightly different...")
        df2 = pd.merge(df[fields], df_attribs, on="name", how='outer')
        df2 = df[fields].groupby(['op_type', 'input_type_shape', 'output_type_shape', 'attribute']).sum()

        df1 = df1.sort_values(by="dur", ascending=False)
        df2 = df2.sort_values(by="dur", ascending=False)
        df1['csum'] = df1['pct'].cumsum()
        df1['avg'] = df1['dur'] / df1['count']
        df2['csum'] = df2['pct'].cumsum()
        df2['avg'] = df2['dur'] / df2['count']

        # print("\n--Check model directories for CSV/JSON outputs--\n--Name:model_traces--\n")
        # print(df1)


    if args.csv:
        df1.to_csv('trace_records_summarized.csv', mode='w+')
        df2.to_csv('trace_records_verbose.csv', mode='w+')

        json_df_Sum = df1.to_json(orient='table', indent=2)
        json_df_Ver = df2.to_json(orient='table', indent=2)
        json_file = open("trace_records_summarized.json", 'w+')
        json_file.write(json_df_Sum)
        json_file.close()
        
        json_file = open("trace_records_verbose.json", 'w+')
        json_file.write(json_df_Ver)
        json_file.close()

if __name__ == '__main__':
    args = get_args()
    logger(args)
