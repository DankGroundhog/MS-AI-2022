import argparse
import os
import shutil
import sys
from timeit import default_timer as timer

import numpy as np
import onnx
import onnxruntime as rt
import psutil
from onnx import helper, numpy_helper


type_dict = {
    'tensor(float16)': np.float16,
    'tensor(float)': np.float32,
    'tensor(double)': np.float64,
    'tensor(int32)': np.int32,
    'tensor(int8)': np.int8,
    'tensor(uint8)': np.uint8,
    'tensor(int16)': np.int16,
    'tensor(uint16)': np.uint16,
    'tensor(int64)': np.int64,
    'tensor(uint64)': np.uint64,
    'tensor(bool)': np.bool8,
}


def fill(shape, dtype, val=1):
    x = np.empty(shape, dtype=type_dict[dtype])
    x.fill(val)
    return x


def get_args():
    parser = argparse.ArgumentParser(description='onnxruntime bench tool')
    parser.add_argument("--model", required=True, help='model path')
    args = parser.parse_args()
    return args


def gen_shape(shape):
    new_shape = []
    for i, s in enumerate(shape):
        if isinstance(s, int):
            new_shape.append(s)
        else:
            if s in ["h", "w"]:
                new_shape.append(224)
            else:
                new_shape.append(1)
    return new_shape


def ort_bench(model):
    opt = rt.SessionOptions()

    opt.enable_profiling = True
    opt.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL
    providers = ['CPUExecutionProvider']

    sess = rt.InferenceSession(model, sess_options=opt, providers=providers)

    feeds = {}
    for input_meta in sess.get_inputs():
        # replace any symbolic dimensions (value is None) with 1
        shape = gen_shape(input_meta.shape)
        dtype = input_meta.type
        name = input_meta.name
        feeds[name] = fill(shape, dtype)

    for i in range(5):
        sess.run([], feeds)  # fetch all outputs

    trace_file = sess.end_profiling()
    target = "trace.json"
    shutil.move(trace_file, target)


if __name__ == "__main__":
    args = get_args()
    ret = ort_bench(args.model)
    sys.exit(0)
