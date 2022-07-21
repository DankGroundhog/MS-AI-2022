import argparse
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

# For each entry in the df, depending on the op_type, input/output shape
# attribute. Maybe padding too???

# def process_and_make(data_df, stats_df):
def process_and_make():
    # Receive data_df and for each entry, create a node.
    #    0          1         2              3                                  4                              5
    # "op_type", "count", 'op_type', 'input_type_shape' [JSON string], 'output_type_shape' [JSON string], 'attribute'
    data_df = pd.read_csv("models/bertsquad-12/model_traces/trace_records_verbose.csv")
    # print(data_df)
    # inputs_to_sort = []
    nodes = []
    input_buffer, output_buffer, op_buffer, attrib_buffer = None, None, None, None
    input_pos = 0 # This var is used for the input/output tags of the nodes
    initial_input = None

    for i in range(len(data_df)):
        buffer_string = ''
        buffer_list = list()
        input_buffer = json.loads(data_df['input_type_shape'][i])

        # if input_buffer[0].keys() == 'int*':
        #     shape_type = int()
        # elif input_buffer[0].keys() == 'float':
        #     shape_type = float()

        output_buffer = json.loads(data_df['output_type_shape'][i])
        attrib_buffer = data_df['attribute'][i]
        
        if list(input_buffer[0].keys())[0].__contains__("int"):
            if attrib_buffer == '[]':
                buffer_list = []
            else:
                buffer_list = attrib_buffer.strip('[]').split(',')
                for i in range(len(buffer_list)):
                    buffer_list[i] = int(buffer_list[i].strip().strip('\''))

            input_buffer = ''.join(str(list(input_buffer[0].values())))
            input_buffer = input_buffer.strip('[]').split(',')
            for item in range(len(input_buffer)): input_buffer[item] = int(input_buffer[item])
            output_buffer = ''.join(str(list(output_buffer[0].values())))
            output_buffer = output_buffer.strip('[]').split(',')
            for item in range(len(output_buffer)): output_buffer[item] = int(output_buffer[item])

        elif list(input_buffer[0].keys())[0] == 'float':
            input_buffer = ''.join(str(list(input_buffer[0].values())))
            input_buffer = input_buffer.strip('[]').split(',')
            for item in range(len(input_buffer)): input_buffer[item] = float(input_buffer[item])
            output_buffer = ''.join(str(list(output_buffer[0].values())))
            output_buffer = output_buffer.strip('[]').split(',')
            for item in range(len(output_buffer)): output_buffer[item] = float(output_buffer[item])

            if attrib_buffer == '[]':
                buffer_list = []
            else:
                if attrib_buffer.find("[[[") == -1:
                    # I know this looks super redundant, but this is how it works consistently trust me
                    attrib_list_dump = []
                    buffer_list = attrib_buffer.strip('[ \' [\' \'] \' ]').split(',')
                    for i in range(0, len(buffer_list), 2):
                        buffer_list[i] = str(buffer_list[i].strip(" [\' \'] "))
                        if i != len(buffer_list):
                            buffer_list[i+1] = float(buffer_list[i+1].strip(" \' [\' \'] \'"))
                    for i in range(0, len(buffer_list), 2):
                        if i != len(buffer_list):
                            buffer_list[i] = [buffer_list[i], buffer_list[i+1]]        
                    for j in range(len(buffer_list)):
                        if isinstance(buffer_list[j], list):
                            attrib_list_dump.append(buffer_list[j])
                    buffer_list = attrib_list_dump
                    del attrib_list_dump
                else:
                    buffer_list = attrib_buffer.strip('[]').split(',')
                    buffer_list[0] = str(buffer_list[0].strip("\'"))                   
                    buffer_list[1] = float(buffer_list[1].strip().strip("\'"))

            # input_buffer = ''.join(str(list(input_buffer[0].values())))
            # input_buffer = input_buffer.strip('[]').split(',')
            # for item in range(len(input_buffer)): input_buffer[item] = float(input_buffer[item])
            # output_buffer = ''.join(str(list(output_buffer[0].values())))
            # output_buffer = output_buffer.strip('[]').split(',')
            # for item in range(len(output_buffer)): output_buffer[item] = float(output_buffer[item])

        attrib_buffer = buffer_list
        del buffer_list
        
        # attrib_buffer = dict(attrib_buffer)
        op_buffer = data_df['op_type'][i]

        # Fix input encoding: Passes 1 string - Model interprets as a list of
        # Convert the node list to a directed graph, then turn that graph into the model?

        # input_buffer = ''.join(str(list(input_buffer[0].values())))
        # input_buffer = input_buffer.strip('[]').split(',')
        # for item in range(len(input_buffer)): input_buffer[item] = int(input_buffer[item])
        # output_buffer = ''.join(str(list(output_buffer[0].values())))
        # output_buffer = output_buffer.strip('[]').split(',')
        # for item in range(len(output_buffer)): output_buffer[item] = int(output_buffer[item])

        # input_buffer = ''.join(str(input_buffer[0].values()))
        # output_buffer = ''.join(str(output_buffer[0].values()))
    
        #kernel shape = attrib
        #input = list of names
        #input name is just a 'pointer'
        #op_buffer + pos number for the input 
        nodes.append(make_node(op_buffer, inputs=str(input_pos), outputs=str(input_pos + 1), kernel_shape=input_buffer, kwargs=attrib_buffer))
        input_pos += 1
        if i == 0: initial_input = input_buffer
        input_buffer, output_buffer, op_buffer, attrib_buffer = None, None, None, None

    # X = np.array([[1, 1, 1], # Graph Input | Make generic by reading first DF sorted by input (small --> large)
    #           [1, 1, 1], 
    #           [1, 1, 1]], dtype=np.float32).reshape(1, 1, 3, 3)

    model = helper.make_model(
    opset_imports=[helper.make_operatorsetid('', 11)],
    producer_name='Synthetic-Model-test',
    graph=make_graph(
        name='test',
        inputs=[helper.make_tensor_value_info('X', TensorProto.FLOAT, shape=initial_input)],
        outputs=[helper.make_tensor_value_info('Y', TensorProto.FLOAT, shape=[1,25])],
        nodes=nodes,
        ),
    )

    onnx.save(model, os.path.join(os.getcwd(), "model.onnx"))
    print(f"output in {os.getcwd()}")
    # for column in ["input_type_shape"]:
    #     for row in range(len(data_df.index)):
    #         buffer = json.loads(data_df[column][row])
            
    #         temp_inputs.extend([list(ele.values()) for ele in buffer])
    #         temp_inputs = []
    # print(temp_inputs)
            

    # for entry in data_df:
    #     if entry == 'input_type_shape':
    #         temp_input = json.loads(data_df[entry][])
         
        # nodes.append(make_node(entry[0], inputs=))

if __name__ == "__main__":
    process_and_make()