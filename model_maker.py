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
import os

import onnx
from onnx import ModelProto, helper, numpy_helper, onnx_pb

def get_args():
    parser = argparse.ArgumentParser(description='onnxruntime modelmaking tool')
    parser.add_argument('--input', required=True, help='input file, whether CSV or JSON')
    parser.add_argument('--name', help='names the synthetic model')
    args = parser.parse_args()
    return args

def define_model():
    exit()

def make_model():
    exit()

if __name__ == "__main__":
    args = get_args()
    exit()