import argparse
from curses.ascii import isdigit
from fnmatch import fnmatch
from importlib.resources import open_binary
import json
import logging
import re
import sys
from attr import attributes
from hamcrest import none
from matplotlib.font_manager import json_load
from matplotlib.pyplot import pause, step

import numpy as np
import pandas as pd
import json
import os
import csv

import tf2onnx

import networkx as nx

import onnx
from onnx import ModelProto, helper, numpy_helper, onnx_pb, AttributeProto, TensorProto, GraphProto
from onnx.helper import make_graph
from _curses import *
import re
from getAttribs import default

def make_node(op_type, inputs, outputs, name=None, doc_string=None, domain=None, **kwargs):
    node = helper.make_node(op_type, inputs, outputs, name, doc_string, domain, **kwargs)
    if doc_string == '':
        node.doc_string = ''
    return node

def save_protobuf(path, message):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "wb") as f:
        f.write(message.SerializeToString())

def get_inputs_and_outputs(main_dir):
    inputs, outputs = [], []
    data, buffer = None, None
    os.chdir(f'{main_dir}')
    for model in os.listdir():
        os.chdir(f'{main_dir}/{model}')
        for file in os.listdir():
            if fnmatch(file, '*.onnx'):
                data = onnx.load(file)
                inputs.append(data.graph.input)
                outputs.append(data.graph.output)
    os.chdir('..')
    return inputs, outputs

def get_attributes_from_buffers(main_dir):
    # Reads attribute string buffer files and extracts the data
    # in the order in which it is organized.

    attribute_list = []
    os.chdir(f'{main_dir}')
    for model in os.listdir():
        os.chdir(f'{main_dir}/{model}/model_traces')
        for file in os.listdir():
            if fnmatch(file, 'attrib_buffer.json'):
                with open(file, 'r') as f:
                    # f = f.read()
                    attribute_list.append([json.load(f)])
                    break
    
    # Returns a list of lists containing a models node names/attributes
    # per each index.
    return attribute_list

def process_and_make(main_dir):
    # Receive data_df and for each entry, create a node. Ideal order of indices
    #    0          1         2              3                                  4                              5
    # "op_type", "count", 'op_type', 'input_type_shape' [JSON string], 'output_type_shape' [JSON string], 'attribute'
    
    # for every verbose csv in --dir "models", do ALL of this

    graph_inputs, graph_outputs = get_inputs_and_outputs(main_dir)
    ops_per_model = get_attributes_from_buffers(main_dir)
    op_attribs = [] # op_names (indices) op_attribs (data)
    for i in range(len(ops_per_model)):
        # op_names.append(list(ops_per_model[i][0].keys()))
        op_attribs.append(list(ops_per_model[i][0].values()))
    # for j in range(len(op_names)):
    #     op_names[0] += op_names[j]
    # if op_names != list():
    #     op_names = op_names[0]
    for k in range(len(op_attribs)):
        op_attribs[0] += op_attribs[k]
    if op_attribs != list():
        op_attribs = op_attribs[0]

    data_df = None
    nodes = []
    input_pos = 0 # This var is used for the input/output tags of the nodes
    # initial_input = None
    graph_input_buffer, graph_output_buffer = [], []
    for i in range(len(graph_inputs)):
        # graph_inputs[i][0] is already a ValueProto object. Only adding it without creating a new TensorValue or ValueProto equivalent. 
        graph_input_buffer.append(graph_inputs[i][0])
    for j in range(len(graph_outputs)):
        # graph_outputs[j][0] is already a ValueProto object. Only adding it without creating a new TensorValue or ValueProto equivalent. 
        graph_output_buffer.append(graph_outputs[j][0])

    for model in os.listdir(main_dir):
        os.chdir(f'{main_dir}/{model}')
        for log in os.listdir('model_traces'):
            if fnmatch(log, '*_verbose.csv'):
    
                data_df = pd.read_csv(f'model_traces/{log}')
                # input_buffer, output_buffer, op_buffer, attrib_buffer = None, None, None, None
                break


        # # Code segment to find the first op that fits one graph input
        # # Actually do it recursively, might be easier than expected
        # count = 0
        # for op in range(len(data_df)):
        #     op_buffer = data_df['op_type'][op]
            
        #     i_o = data_df["I/O"][op].split(',')
        #     input = str(i_o[0].strip("\' [] \'"))
        #     output = str(i_o[1].strip("\' [] \'"))
        #     if op == 0:
        #         for node in range(len(data_df)):
        #             i_o = data_df["I/O"][node].split(',')
        #             input = str(i_o[0].strip("\' [] \'"))
        #             output = str(i_o[1].strip("\' [] \'"))
        #             for item in range(len(graph_input_buffer)):
        #                 if input == graph_input_buffer[item].name:
        #                     nodes.append(make_node(op_buffer, inputs=[input], outputs=[output], name=data_df["name"][i], attributes=data_df["attribute"][i]))
        #                     count += 1
        #     # Could have more than one op consuming the graph inputs jumps to the next node
        #     # after the ones already in the node list
        #     j = i + count
        #     for j in range(len(nodes)):
        #         prev_out = nodes[j-1].output
        #         if prev_out[0] == input:
        #             nodes.append(make_node(op_buffer, inputs=[input], outputs=[output], name=data_df["name"][j], attributes=data_df["attribute"][j]))

        for i in range(len(data_df)):
            op_buffer = data_df['op_type'][i]
            
            i_o = data_df["I/O"][i].split(',')
            input = str(i_o[0].strip("\' [] \'"))
            output = str(i_o[1].strip("\' [] \'"))
            attribute_obj = json.loads(data_df["attribute"][i].replace('\'', '\"'))
            # attr_name = list(attribute_obj.keys())[0]
            nodes.append(make_node(op_buffer, inputs=[input], outputs=[output], name=data_df["name"][i], **json.loads(data_df["attribute"][i].replace('\'', '\"'))))
            
            # New Code segment with arrangement logic down here
            # i_o = data_df["I/O"][i].split(',')
            # input = str(i_o[0].strip("\' [] \'"))
            # output = str(i_o[1].strip("\' [] \'"))
            # if i == 0:
            #     j = i
            #     for j in range(len(graph_input_buffer)):
            #         i_o = data_df["I/O"][i].split(',')
            #         input = str(i_o[0].strip("\' [] \'"))
            #         output = str(i_o[1].strip("\' [] \'"))
            #         if input == graph_input_buffer[j]:
            #             nodes.append(make_node(op_buffer, inputs=[input], outputs=[output], name=data_df["name"][i], attributes=data_df["attribute"][i]))
    
    # nodes = tf2onnx.graph_builder
    os.chdir(f'..')

    model = helper.make_model(
    opset_imports=[helper.make_operatorsetid('', 11)],
    producer_name='Synthetic-Model-test',
    graph=make_graph(
        name='test',
        # inputs=[helper.make_tensor_value_info('A', TensorProto.FLOAT, shape=initial_input)],
        inputs=graph_input_buffer,
        outputs=graph_output_buffer,
        # outputs=[helper.make_tensor_value_info('Y', TensorProto.FLOAT, shape=[1,25])],
        nodes=nodes,
        ),
    )
    # model.graph.topological_sort(nodes)
    onnx.save(model, os.path.join(os.getcwd(), "synthetic_model.onnx"))
    print(f"Output in {os.getcwd()}")

if __name__ == "__main__":
    process_and_make()