# Created By: Felix M. Perez Quinones, AI Platform, ONNX (2022 Internship)
# Sourced from different scripts in the ONNX repo, modified for convenience

# Can be used to debug if run by itself or as an imported function, which
# is the current use.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
from onnx import ModelProto, helper, numpy_helper, onnx_pb

from six import text_type, integer_types, binary_type

from onnx import ModelProto, helper, numpy_helper, onnx_pb, AttributeProto, TensorProto, GraphProto

from onnx.mapping import STORAGE_TENSOR_TYPE_TO_FIELD
from typing import Text, Sequence, TypeVar, Callable

import pandas as pd

import json

attrib_json = None

def set_attrib_json():
    return attrib_json

def default(o):
    try:
        iterable = iter(o)
    except TypeError:
        raise TypeError(f'Object of type {o.__class__.__name__} '
                        f'is not JSON serializable')
    else:
        return list(iterable)
        # Let the base class default method raise the TypeError
    
    # return JSONEncoder.default(, o)

def attribs(model_proto):

    def str_float(f):  # type: (float) -> Text
        # NB: Different Python versions print different numbers of trailing
        # decimals, specifying this explicitly keeps it consistent for all
        # versions
        return '{:.15g}'.format(f)

    def str_int(i):  # type: (int) -> Text
        # NB: In Python 2, longs will repr() as '2L', which is ugly and
        # unnecessary.  Explicitly format it to keep it consistent.
        return '{:d}'.format(i)

    def str_str(s):  # type: (Text) -> Text
        return repr(s)

    _T = TypeVar('_T')  # noqa

    def str_list(str_elem, xs):  # type: (Callable[[_T], Text], Sequence[_T]) -> Text
        return '[' + ', '.join(map(str_elem, xs)) + ']'
    
    def _sanitize_str(s):  # type: (Union[Text, bytes]) -> Text
        if isinstance(s, text_type):
            sanitized = s
        elif isinstance(s, binary_type):
            sanitized = s.decode('utf-8', errors='ignore')
        else:
            sanitized = str(s)
        if len(sanitized) < 64:
            return sanitized
        return sanitized[:64] + '...<+len=%d>' % (len(sanitized) - 64)
    
    content = []
    node_data = {}
    attribs = {}
    for node in model_proto.graph.node:
        # attribs = []
        node_data['name'] = node.name
        node_data['I/O'] = [node.input[0], node.output[0]]
        # print(node.input)
        for attr in node.attribute:
            if attr.HasField("f"):
                attribs[attr.name] = attr.f
            elif attr.HasField("i"):
                attribs[attr.name] = attr.i
            elif attr.HasField("s"):
                # TODO: Bit nervous about Python 2 / Python 3 determinism implications
                attribs[attr.name] = repr(_sanitize_str(attr.s))
            elif attr.HasField("t"):
                if len(attr.t.dims) > 0:
                    attribs[attr.t.name] = "<Tensor>"
                else:
                    # special case to print scalars
                    field = STORAGE_TENSOR_TYPE_TO_FIELD[attr.t.data_type]
                    attribs[attr.t.name] = getattr(attr.t, field)
            elif attr.HasField("g"):
                attribs[attr.name] = attr.g.name
            elif attr.HasField("tp"):
                attribs[attr.name] = attr.tp
            elif attr.floats:
                attribs[attr.name] = attr.floats
            elif attr.ints:
                attribs[attr.name] = attr.ints
            elif attr.strings:
                # TODO: Bit nervous about Python 2 / Python 3 determinism implications
                attribs[attr.name] = list(map(_sanitize_str, attr.strings))
            elif attr.tensors:
                attribs[attr.name] = "[<Tensor>, ...]"
            # Fix this to fit the dictionary fix
            elif attr.type_protos:
                attribs.append('[')
                for i, tp in enumerate(attr.type_protos):
                    comma = ',' if i != len(attr.type_protos) - 1 else ''
                    attribs.append('<Type Proto {}>{}'.format(tp, comma))
                attribs.append(']')
            # Fix this to fit the dictionary fix
            elif attr.graphs:
                attribs.append('[')
                for i, g in enumerate(attr.graphs):
                    comma = ',' if i != len(attr.graphs) - 1 else ''
                    attribs.append('<graph {}>{}'.format(g.name, comma))
                attribs.append(']')
            else:
                attribs[attr.name] = "<Unknown>"
        node_data['attribute'] = attribs
        content.append(json.dumps(node_data, default=default))
        attribs = {}
    
    buffer_dict = dict()
    for i in range(len(content)):
        # For every element, json.loads into the dict
        buffer_dict[f'{i}'] = json.loads(content[i])

    attrib_json_file = open("attrib_buffer.json", "w+")
    json.dump(buffer_dict, attrib_json_file, indent=3)

    return pd.DataFrame(buffer_dict).transpose()

if __name__ == '__main__':
    with open("bertsquad-12.onnx", "rb") as f: # Code for testing, not for running it individually
        data = f.read()
        model_proto = ModelProto()
        model_proto.ParseFromString(data)

    def str_float(f):  # type: (float) -> Text
        # NB: Different Python versions print different numbers of trailing
        # decimals, specifying this explicitly keeps it consistent for all
        # versions
        return '{:.15g}'.format(f)

    def str_int(i):  # type: (int) -> Text
        # NB: In Python 2, longs will repr() as '2L', which is ugly and
        # unnecessary.  Explicitly format it to keep it consistent.
        return '{:d}'.format(i)

    def str_str(s):  # type: (Text) -> Text
        return repr(s)

    _T = TypeVar('_T')  # noqa

    def str_list(str_elem, xs):  # type: (Callable[[_T], Text], Sequence[_T]) -> Text
        return '[' + ', '.join(map(str_elem, xs)) + ']'
    
    def _sanitize_str(s):  # type: (Union[Text, bytes]) -> Text
        if isinstance(s, text_type):
            sanitized = s
        elif isinstance(s, binary_type):
            sanitized = s.decode('utf-8', errors='ignore')
        else:
            sanitized = str(s)
        if len(sanitized) < 64:
            return sanitized
        return sanitized[:64] + '...<+len=%d>' % (len(sanitized) - 64)
    
    content = []
    node_data = {}
    attribs = {}
    for node in model_proto.graph.node:
        node_data['name'] = node.name
        for attr in node.attribute:
            if attr.HasField("f"):
                attribs[attr.name] = attr.f
            elif attr.HasField("i"):
                attribs[attr.name] = attr.i
            elif attr.HasField("s"):
                # TODO: Bit nervous about Python 2 / Python 3 determinism implications
                attribs[attr.name] = repr(_sanitize_str(attr.s))
            elif attr.HasField("t"):
                if len(attr.t.dims) > 0:
                    attribs[attr.t.name] = "<Tensor>"
                else:
                    # special case to print scalars
                    field = STORAGE_TENSOR_TYPE_TO_FIELD[attr.t.data_type]
                    attribs[attr.t.name] = getattr(attr.t, field)
            elif attr.HasField("g"):
                attribs[attr.name] = attr.g.name
            elif attr.HasField("tp"):
                attribs[attr.name] = attr.tp
            elif attr.floats:
                attribs[attr.name] = attr.floats
            elif attr.ints:
                attribs[attr.name] = attr.ints
            elif attr.strings:
                # TODO: Bit nervous about Python 2 / Python 3 determinism implications
                attribs[attr.name] = list(map(_sanitize_str, attr.strings))
            elif attr.tensors:
                attribs[attr.name] = "[<Tensor>, ...]"
            # Fix this to fit the dictionary fix
            elif attr.type_protos:
                attribs.append('[')
                for i, tp in enumerate(attr.type_protos):
                    comma = ',' if i != len(attr.type_protos) - 1 else ''
                    attribs.append('<Type Proto {}>{}'.format(tp, comma))
                attribs.append(']')
            # Fix this to fit the dictionary fix
            elif attr.graphs:
                attribs.append('[')
                for i, g in enumerate(attr.graphs):
                    comma = ',' if i != len(attr.graphs) - 1 else ''
                    attribs.append('<graph {}>{}'.format(g.name, comma))
                attribs.append(']')
            else:
                attribs[attr.name] = "<Unknown>"

        node_data['attribute'] = attribs
        content.append(json.dumps(node_data, default=default))

        attribs = {}
    
    buffer_dict = dict()
    for i in range(len(content)):
        # For every element, json.loads into the dict
        buffer_dict[f'{i}'] = json.loads(content[i])

    attrib_json_file = open("attrib_buffer.json", "w+")
    json.dump(buffer_dict, attrib_json_file, indent=3)
    print(pd.DataFrame(buffer_dict).transpose())