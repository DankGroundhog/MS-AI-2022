import argparse
from importlib.resources import open_binary
import json
import logging
import re
import sys
from attr import attributes
from hamcrest import none
from matplotlib.font_manager import json_load
from matplotlib.pyplot import pause

import numpy as np
import pandas as pd
import json
import os
import csv

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
    inputs_to_sort = []
    nodes = []
    input_buffer, output_buffer, op_buffer = None, None, None

    for i in range(len(data_df)):
        input_buffer = json.loads(data_df['input_type_shape'][i])
        output_buffer = json.loads(data_df['output_type_shape'][i])
        op_buffer = data_df['op_type'][i]

        # Fix input encoding: Passes 1 string - Model interprets as a list of
        # Convert the node list to a directed graph, then turn that graph into the model?

        input_buffer = ''.join(str(list(input_buffer[0].values())))
        output_buffer = ''.join(str(list(output_buffer[0].values())))
    

        nodes.append(make_node(op_buffer, inputs=input_buffer, outputs=output_buffer, name=None, doc_string=None, domain=None))
        input_buffer, output_buffer, op_buffer = None, None, None

    X = np.array([[1, 1, 1], # Graph Input | Make generic by reading first DF sorted by input (small --> large)
              [1, 1, 1], 
              [1, 1, 1]], dtype=np.float32).reshape(1, 1, 3, 3)

    model = helper.make_model(
    opset_imports=[helper.make_operatorsetid('', 11)],
    producer_name='Synthetic-Model-test',
    graph=make_graph(
        name='test',
        inputs=[helper.make_tensor_value_info('X', TensorProto.FLOAT, shape=X.shape)],
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