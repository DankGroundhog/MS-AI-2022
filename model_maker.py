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

import networkx as nx

import onnx
from onnx import ModelProto, helper, numpy_helper, onnx_pb, AttributeProto, TensorProto, GraphProto
from onnx.helper import make_graph
from _curses import *
import re

# def get_args():
#     parser = argparse.ArgumentParser(description='onnxruntime modelmaking tool')
#     parser.add_argument('--input', required=True, help='input file, whether CSV or JSON')
#     parser.add_argument('--name', help='names the synthetic model')
#     args = parser.parse_args()
#     return args

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

def process_and_make(main_dir):
    # Receive data_df and for each entry, create a node. Ideal order of indices
    #    0          1         2              3                                  4                              5
    # "op_type", "count", 'op_type', 'input_type_shape' [JSON string], 'output_type_shape' [JSON string], 'attribute'
    
    # for every verbose csv in --dir "models", do ALL of this

    graph_inputs, graph_outputs = get_inputs_and_outputs(main_dir)

    nodes = []
    input_pos = 0 # This var is used for the input/output tags of the nodes
    # initial_input = None
    for model in os.listdir(main_dir):
        os.chdir(f'{main_dir}/{model}')
        for log in os.listdir('model_traces'):
            if fnmatch(log, '*_verbose.csv'):
    
                data_df = pd.read_csv(f'model_traces/{log}')
                # nodes = []
                input_buffer, output_buffer, op_buffer, attrib_buffer = None, None, None, None
                # input_pos = 0 # This var is used for the input/output tags of the nodes
                # initial_input = None

                for i in range(len(data_df)):
                    buffer_string = ''
                    buffer_list = list()
                    input_buffer = json.loads(data_df['input_type_shape'][i])

                    output_buffer = json.loads(data_df['output_type_shape'][i])
                    attrib_buffer = data_df['attribute'][i]
                    
                    if list(input_buffer[0].keys())[0].__contains__("int"):
                        if attrib_buffer == '[]':
                            buffer_list = []
                        else:
                            buffer_list = attrib_buffer.strip('[]').split(',')
                            for i in range(len(buffer_list)):
                                if type(buffer_list[i]) == type(str()) and re.search('[a-zA-Z]+', buffer_list[i]):
                                        buffer_list[i] = str(buffer_list[i].strip(" [\' \'] "))
                                else:
                                    buffer_list[i] = int(buffer_list[i].strip(" \' [\' \'] \'"))

                        input_buffer = ''.join(str(list(input_buffer[0].values())))
                        input_buffer = input_buffer.strip('[]').split(',')
                        if input_buffer[0] != '':
                            for item in range(len(input_buffer)): input_buffer[item] = int(input_buffer[item])
                        output_buffer = ''.join(str(list(output_buffer[0].values())))
                        output_buffer = output_buffer.strip('[]').split(',')
                        if output_buffer[0] != '':
                            for item in range(len(output_buffer)): output_buffer[item] = int(output_buffer[item])

                    elif list(input_buffer[0].keys())[0] == 'float':
                        input_buffer = ''.join(str(list(input_buffer[0].values())))
                        input_buffer = input_buffer.strip('[]').split(',')
                        if input_buffer[0] != '':
                            for item in range(len(input_buffer)): input_buffer[item] = float(input_buffer[item])
                        output_buffer = ''.join(str(list(output_buffer[0].values())))
                        output_buffer = output_buffer.strip('[]').split(',')

                        if output_buffer[0] != '':
                            for item in range(len(output_buffer)): output_buffer[item] = float(output_buffer[item])

                        if attrib_buffer == '[]':
                            buffer_list = []
                        else:
                            if attrib_buffer.find("[[[") == -1:
                                # I know this looks super redundant, but this is how it works consistently, trust me ':)
                                attrib_list_dump, name, attr_vals = [], None, []
                                buffer_list = attrib_buffer.strip(" [\' \'] ").strip('[ \' [\' \'] \' ]').split(',')
                                for i in range(len(buffer_list)):
                                    # If it is not a digit, string, else, corresponding datatype
                                    if type(buffer_list[i]) == type(str()) and re.search('[a-zA-Z]+', buffer_list[i]):
                                        buffer_list[i] = str(buffer_list[i].strip(" [\' \'] "))
                                    else:
                                        buffer_list[i] = float(buffer_list[i].strip(" \' [\' \'] \'"))
                                # if it is a string, assign to name, else, assign to attr_vals. Then merge again in attrib_list
                                i = 0
                                while i != (len(buffer_list)):
                                    if type(buffer_list[i]) == type(str()):
                                        name = buffer_list[i]
                                    else:
                                        if not (i >= len(buffer_list) - 1) and type(buffer_list[i+1]) != type(str()):
                                            attr_vals.append(buffer_list[i])
                                        else:
                                            attr_vals.append(buffer_list[i])
                                            attrib_list_dump.append([name, attr_vals])
                                            attr_vals = []
                                    i += 1

                                buffer_list = attrib_list_dump
                                del attrib_list_dump, name
                            else:
                                buffer_list = attrib_buffer.strip('[]').split(',')
                                buffer_list[0] = str(buffer_list[0].strip("\'"))                   
                                buffer_list[1] = float(buffer_list[1].strip().strip("\'"))

                    attrib_buffer = buffer_list
                    del buffer_list

                    op_buffer = data_df['op_type'][i]

                    # Fix input encoding: Passes 1 string - Model interprets as a list of characters
                    # Convert the node list to a directed graph, then turn that graph into the model?
                    
                    #Adding kernel_hape to the attribute list
                    attrib_buffer = [["attributes", attrib_buffer], ["kernel_shape", input_buffer]]
                    attrib_buffer = dict(attrib_buffer)
                    # nodes.append(make_node(op_buffer, inputs=str(input_pos), outputs=str(input_pos + 1), name=str(input_pos), kernel_shape=input_buffer, kwargs=attrib_buffer))
                    nodes.append(make_node(op_buffer, inputs=[str(input_pos)], outputs=[str(input_pos + 1)], name=str(input_pos), kwargs=attrib_buffer))
                    input_pos += 1
                    # if i == 0: initial_input = input_buffer
                    input_buffer, output_buffer, op_buffer, attrib_buffer = None, None, None, None

                # X = np.array([[1, 1, 1], # Graph Input | Make generic by reading first DF sorted by input (small --> large)
                #           [1, 1, 1], 
                #           [1, 1, 1]], dtype=np.float32).reshape(1, 1, 3, 3)

    os.chdir(f'..')
    graph_input_buffer, graph_output_buffer = [], []
    for i in range(len(graph_inputs)):
        # graph_inputs[i][0] is already a ValueProto object. Only adding it without creating a new TensorValue or ValueProto equivalent. 
        graph_input_buffer.append(graph_inputs[i][0])
    for j in range(len(graph_outputs)):
        # graph_outputs[j][0] is already a ValueProto object. Only adding it without creating a new TensorValue or ValueProto equivalent. 
        graph_output_buffer.append(graph_outputs[j][0])

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

    onnx.save(model, os.path.join(os.getcwd(), "synthetic_model.onnx"))
    print(f"Output in {os.getcwd()}")

if __name__ == "__main__":
    process_and_make()