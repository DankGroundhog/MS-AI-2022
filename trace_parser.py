'''
trace_parser takes a directory where models are stored, it then creates a trace per model (naming the trace),
to then use the ort-trace.py and record the output into a CSV and JSON, in table format.
'''

# Created By: Felix M. Perez Quinones, AI Platform, ONNX (2022 Internship)


import os, sys, numpy as np, argparse, shutil, onnx, fnmatch
from posixpath import dirname
from time import sleep
from stats_calculator import calculator
from model_maker import process_and_make

def get_args():
    '''
    Arg list:
        --dir [directory path(folder name if trace_parser is in the same directory)]
        -v [verbose stats logging/printing for model creation]
    '''   
    parser = argparse.ArgumentParser(description='Look at the code for flag descriptions')
    parser.add_argument("-d","--dir", required=True, help='directory path or folder name where model.onnx(s) are located')
    parser.add_argument("-v", required=False, action='store_true', help="verbose stat logging for model creation")
    args = parser.parse_args()
    return args        

def model_dir_reader(dir_name, main_dir, args):
    # Read models from a directory where, ideally, .onnx files are stored. 
    # Runs ortperf.py, gets a trace, renames the trace to the AI/ML model name 
    # and stores it into a folder (create the folder) where it can then be processed. 


    if os.path.exists(dir_name) and os.path.isdir(dir_name):
        for model in os.listdir(dir_name):
            os.chdir(f'{main_dir}\{dir_name}\{model}')
            for filename in os.listdir():
                if fnmatch.fnmatch(filename, '*.onnx') and not fnmatch.fnmatch(filename, 'model*'):
                    print(f"Currently processing: {filename}...")
                    os.system(f"python ..\..\ortperf.py --model {filename}")
                    json_name = fnmatch.filter(os.listdir(),'*.json')
                    os.rename(json_name[0], f'{model}_trace.json')
                    if os.path.exists('model_traces'):
                        shutil.rmtree('model_traces')
                        os.mkdir('model_traces')
                    else:
                        os.mkdir('model_traces')
                    shutil.move(f'{model}_trace.json', 'model_traces')
                    os.chdir('model_traces')
                    sub_filename = os.listdir()[0]
                    if fnmatch.fnmatch(sub_filename, '*.json'):
                        os.system(f"python ..\..\..\ort_trace.py --input {sub_filename} -v --csv --source ../{filename}")   
                    print(f"{filename} processing: DONE")
                    break
                elif fnmatch.fnmatch(filename, '*.onnx'):
                    os.rename(filename, f'{model}.onnx')
                    print(f"Currently processing: {filename}...")
                    os.system(f"python ..\..\ortperf.py --model {filename}")
                    json_name = fnmatch.filter(os.listdir(),'*.json')
                    os.rename(json_name[0], f'{model}_trace.json')
                    if os.path.exists('model_traces'):
                        shutil.rmtree('model_traces')
                        os.mkdir('model_traces')
                    else:
                        os.mkdir('model_traces')
                    shutil.move(f'{model}_trace.json', 'model_traces')
                    os.chdir('model_traces')
                    sub_filename = os.listdir()[0]
                    if fnmatch.fnmatch(sub_filename, '*.json'):
                        os.system(f"python ..\..\..\ort_trace.py --input {sub_filename} -v --csv --source ../{filename}")   
                    print(f"{filename} processing: DONE")
                    break
                    # return "model_traces", main_dir
    
    # FIX THIS BRUH
    elif os.path.exists(dir_name) and os.path.isfile(dir_name):
        if fnmatch.fnmatch(dir_name, '*.onnx'):
            print(f"Currently processing: {filename}...")
            os.system(f"python {main_dir}\ortperf.py --model {filename}")
            json_name = fnmatch.filter(os.listdir(),'*.json')
            os.rename(json_name[0], f'{model}_trace.json')
            if os.path.exists('model_traces'):
                shutil.rmtree('model_traces')
                os.mkdir('model_traces')
            else:
                os.mkdir('model_traces')
            shutil.move(f'{model}_trace.json', 'model_traces')
            print(f"{filename} processing: DONE")

    os.chdir(f'{main_dir}')
    calculator(os.getcwd(), args.dir, args.v)
    process_and_make(f'{main_dir}/{args.dir}')
    # Import process_and_make and use it here.


def main():
    args = get_args()
    model_dir_reader(args.dir, os.getcwd(), args)
    return

if __name__ == "__main__":
    main()