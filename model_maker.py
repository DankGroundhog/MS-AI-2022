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
from pandas import Series

import onnx_graphsurgeon as gs # Must download (python[3] -m pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com)
# python[3] -m pip install colored (This one is also needed)

import onnx
from onnx import ModelProto, helper, numpy_helper, onnx_pb, AttributeProto, TensorProto, GraphProto
from onnx.helper import make_graph
from _curses import *
import re
from getAttribs import default

# Maybe creating an adjacency list will produce the correct
# graph for the model.
def create_adjacency_list(node_list, graph_inputs):
    adj_list = {}
    temp = []
    for node in node_list:
        
        continue

    return adj_list

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
                inputs.extend(data.graph.input)
                outputs.extend(data.graph.output)
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
        graph_input_buffer.append(graph_inputs[i])
    for j in range(len(graph_outputs)):
        # graph_outputs[j][0] is already a ValueProto object. Only adding it without creating a new TensorValue or ValueProto equivalent. 
        graph_output_buffer.append(graph_outputs[j])

    for model in os.listdir(main_dir):
        os.chdir(f'{main_dir}/{model}')
        for log in os.listdir('model_traces'):
            if fnmatch(log, '*_verbose.csv'):
    
                data_df = pd.read_csv(f'model_traces/{log}')
                # input_buffer, output_buffer, op_buffer, attrib_buffer = None, None, None, None
                break

        for i in range(len(data_df)):
            op_buffer = data_df['op_type'][i]
            
            i_o = data_df["I/O"][i].split(',')
            input = str(i_o[0].strip("\' [] \'"))
            output = str(i_o[1].strip("\' [] \'"))
            # attribute_obj = json.loads(data_df["attribute"][i].replace('\'', '\"'))
            # attr_name = list(attribute_obj.keys())[0]
            i_shape_type = dict()
            o_shape_type = dict()
            try:
                # if not isinstance(type(data_df['input_type_shape'][i]), Series):
                # print(type(data_df['input_type_shape'][i]))
                if isinstance(data_df['input_type_shape'][i], Series):
                    input_type_shape = json.loads(data_df['input_type_shape'][i].strip("[]"))
                else:
                    input_type_shape = json.loads(data_df['input_type_shape'][i].to_string().strip("[]"))
            except AttributeError:
                input_type_shape = data_df['input_type_shape'][i].strip("[]").replace("{", '').replace("}", '').replace("\'", "\"")
                input_type_shape = "{" + input_type_shape + "}"
                input_type_shape = json.loads(input_type_shape)
                # i_shape_type['input'] = input_type_shape
                # input_type_shape = dict()
                # input_type_shape = i_shape_type
            except json.decoder.JSONDecodeError:
                # input_type_shape = data_df['input_type_shape'].strip("[]").replace("{", '').replace("}", '').replace("\"", "\'")
                input_type_shape = data_df['input_type_shape'].strip("[]").replace("{", '').replace("}", '').replace("\'", "\"")
                input_type_shape = "{" + input_type_shape + "}"
                input_type_shape = json.loads(input_type_shape)
                # for j in range(len(input_type_shape)):
                #     if '}' not in input_type_shape[j]:
                #         input_type_shape[j] += '}'
                #         input_type_shape[j] = input_type_shape[j].strip()
                # input_type_shape.pop(len(input_type_shape)-1)
                # for j in range(len(input_type_shape)):
                #     i_shape_type[f'input_{j+1}'] = json.loads(input_type_shape[j])
                # input_type_shape = input_type_shape
            try:
                # print(type(data_df['output_type_shape'][i]))
                if isinstance(data_df['output_type_shape'][i], Series):
                    output_type_shape = json.loads(data_df['output_type_shape'][i].strip("[]"))
                else:
                    output_type_shape = json.loads(data_df['output_type_shape'][i].to_string().strip("[]"))
                # o_shape_type['input'] = output_type_shape
                # output_type_shape = dict()
                # output_type_shape = o_shape_type
            except AttributeError:
                output_type_shape = data_df['output_type_shape'][i].strip("[]").replace("{", '').replace("}", "").replace("\'", "\"")
                output_type_shape = "{" + output_type_shape + "}"
                output_type_shape = json.loads(output_type_shape)
            except json.decoder.JSONDecodeError:
                output_type_shape = data_df['output_type_shape'].strip("[]").replace("{", '').replace("}", "").replace("\'", "\"")
                output_type_shape = "{" + output_type_shape + "}"
                output_type_shape = json.loads(output_type_shape)
                # for j in range(len(output_type_shape)):
                #     if '}' not in output_type_shape[j]:
                #         output_type_shape[j] += '}'
                #         output_type_shape[j] = output_type_shape[j].strip()
                # output_type_shape.pop(len(input_type_shape)-1)
                # for j in range(len(input_type_shape)):
                #     o_shape_type[f'output_{j+1}'] = json.loads(output_type_shape[j])
                # output_type_shape = o_shape_type

            if len(input_type_shape.keys()) > 1:
                count = 1
                for key in list(input_type_shape.keys()):
                    input_type_shape[f'input_{count}_shape'] = input_type_shape.pop(key)
                    # del input_type_shape[key]
                    count += 1
            else:
                input_type_shape['input_shape'] = input_type_shape[list(input_type_shape.keys())[0]]
                del input_type_shape[list(input_type_shape.keys())[0]]
            
            if len(output_type_shape.keys()) > 1:
                count = 1
                for key in list(output_type_shape.keys()):
                    output_type_shape[f'output_{count}_shape'] = output_type_shape.pop(key)
                    # del output_type_shape[key]
                    count += 1
            else:
                output_type_shape['output_shape'] = output_type_shape[list(output_type_shape.keys())[0]]
                del output_type_shape[list(output_type_shape.keys())[0]]

            # print(f'{op_buffer}\n{input}\n{output}\n{data_df["name"][i]}\n{data_df["attribute"][i]}\n{input_type_shape}\n{output_type_shape}\n-----------------------')
            nodes.append(make_node(op_buffer, inputs=[input], outputs=[output], name=data_df["name"][i], **json.loads(data_df["attribute"][i].replace('\'', '\"')), **input_type_shape, **output_type_shape))
            # nodes.append(make_node(op_buffer, name=data_df["name"][i], **json.loads(data_df["attribute"][i].replace('\'', '\"')), **input_type_shape, **output_type_shape))


    del i_o, data_df, input_pos
    # Trying the dictionary method of building the graph
    # Filling up a dictionary where the keys are the inputs
    # and the value is a list of operators that take that input
    input_dict = dict()
    for i in range(len(nodes)):
        if not nodes[i].input[0] in input_dict:
            input_dict[f'{nodes[i].input[0]}'] = nodes[i]
        else:
            if nodes[i].input[0] in input_dict and not isinstance(input_dict[f'{nodes[i].input[0]}'], list):
                input_dict[f'{nodes[i].input[0]}'] = [input_dict[f'{nodes[i].input[0]}']]
                input_dict[f'{nodes[i].input[0]}'].append(nodes[i])
            else:
                input_dict[f'{nodes[i].input[0]}'].append(nodes[i])
    
    # 0 - Regular attributes
    # 1 - input shape
    # 2 - output shape
    input_shape_dict = dict()
    for i in range(len(nodes)):
        if not str(nodes[i].attribute[1].ints) in input_shape_dict:
            input_shape_dict[f'{nodes[i].attribute[1].ints}'] = nodes[i]
        else:
            if str(nodes[i].attribute[1].ints) in input_shape_dict and not isinstance(input_shape_dict[f'{nodes[i].attribute[1].ints}'], list):
                input_shape_dict[f'{nodes[i].attribute[1].ints}'] = [input_shape_dict[f'{nodes[i].attribute[1].ints}']]
                input_shape_dict[f'{nodes[i].attribute[1].ints}'].append(nodes[i])
            else:
                input_shape_dict[f'{nodes[i].attribute[1].ints}'].append(nodes[i])
    # Create a dictionary with shapes too so that if output
    # does not match input (or does not exist) it can string
    # up nodes based on input type and shape
    
    # Creating another list of nodes in the order in which they 
    # take inputs based on input/output relationships
    graph_nodes = []
    for i in range(len(nodes)):
        # Finds the node that satisfies the graph input
        if i == 0:
            for j in range(len(graph_input_buffer)):
                if graph_input_buffer[j].name in input_dict:
                    graph_nodes.append(input_dict[graph_input_buffer[j].name][0])
        # Overall nodes
        else:
            # 0 - Regular attributes
            # 1 - input shape
            # 2 - output shape

            # If previous output fits any next node, add
            # if graph_nodes[i-1].output[0] in input_dict and len(graph_input_buffer) != 0:
            try:
                if graph_nodes[i-1].output[0] in input_dict and input_dict[graph_nodes[i-1].output[0]] != []:
                    graph_nodes.append(input_dict[graph_nodes[i-1].output[0]][0])
                    input_dict[graph_nodes[i-1].output[0]].pop()
                    # input_shape_dict[]
                # If it does not, check for similar shape and add
                elif str(graph_nodes[i-1].attribute[len(graph_nodes[i-1].attribute._values)-1].ints) in input_shape_dict and input_shape_dict[str(graph_nodes[i-1].attribute[len(graph_nodes[i-1].attribute._values)-1].ints)] != []:
                    graph_nodes.append(input_shape_dict[str(graph_nodes[i-1].attribute[len(graph_nodes[i-1].attribute._values)-1].ints)][0])
                    input_shape_dict[str(graph_nodes[i-1].attribute[len(graph_nodes[i-1].attribute._values)-1].ints)].pop()
            except IndexError:
                    if graph_nodes[(i-(i-len(graph_nodes)))-1].output[0] in input_dict and input_dict[graph_nodes[(i-(i-len(graph_nodes)))-1].output[0]] != []:
                        graph_nodes.append(input_dict[graph_nodes[(i-(i-len(graph_nodes)))-1].output[0]][0])
                        input_dict[graph_nodes[(i-(i-len(graph_nodes)))-1].output[0]].pop()
                    # input_shape_dict[]
                    # If it does not, check for similar shape and add
                    elif str(graph_nodes[(i-(i-len(graph_nodes)))-1].attribute[len(graph_nodes[(i-(i-len(graph_nodes)))-1].attribute._values)-1].ints) in input_shape_dict and input_shape_dict[str(graph_nodes[(i-(i-len(graph_nodes)))-1].attribute[len(graph_nodes[(i-(i-len(graph_nodes)))-1].attribute._values)-1].ints)] != []:
                        graph_nodes.append(input_shape_dict[str(graph_nodes[(i-(i-len(graph_nodes)))-1].attribute[len(graph_nodes[(i-(i-len(graph_nodes)))-1].attribute._values)-1].ints)][0])
                        input_shape_dict[str(graph_nodes[(i-(i-len(graph_nodes)))-1].attribute[len(graph_nodes[(i-(i-len(graph_nodes)))-1].attribute._values)-1].ints)].pop()

                

    os.chdir(f'..')

    model = helper.make_model(
    opset_imports=[helper.make_operatorsetid('', 14)],
    producer_name='Synthetic-Model-test',
    graph=make_graph(
        name='test',
        # inputs=[helper.make_tensor_value_info('A', TensorProto.FLOAT, shape=initial_input)],
        inputs=graph_input_buffer,
        outputs=graph_output_buffer,
        # outputs=[helper.make_tensor_value_info('Y', TensorProto.FLOAT, shape=[1,25])],
        nodes=graph_nodes,
        ),
    )

    onnx.save(model, os.path.join(os.getcwd(), "synthetic_model.onnx"))
    # model = gs.import_onnx(onnx.load("synthetic_model.onnx"))
    # os.remove("synthetic_model.onnx")
    # model = model.toposort()
    # # os.remove("synthetic_model.onnx")
    # onnx.save(gs.export_onnx(model), "synthetic_model.onnx")
    print(f"Output in {os.getcwd()}")

    # graph = Graph(len(nodes))
    # graph.graph = model.graph

if __name__ == "__main__":
    process_and_make()