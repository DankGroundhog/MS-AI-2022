'''
trace_parser takes a directory where models are stored, it then creates a trace per model (naming the trace),
to then use the ort-trace.py and record the output into a CSV and JSON, in table format.

TO-DO:
1. Get OP attributes from .onnx file
2. Polish and generalize parser
'''

# Created By: Felix M. Perez Quinones, AI Platform, ONNX (2022 Internship)


import os, sys, numpy as np, argparse, shutil, onnx, fnmatch
from time import sleep


def trace_dir_reader(dir_name, main_dir):
    # Read directory where traces are located, ideally named by model_folder_reader.
    # Takes the folder, runs ort-trace on it, ideally prints terminal output into a DF
    # and then converts DF into CSV/JSON in the same trace directory.

    '''
        DEPRECATED : Combined into model_dir_reader()
    '''

    # Check if directory exists
    if os.path.exists(dir_name):
        os.chdir(dir_name)
        for filename in os.listdir(os.getcwd()):
            os.system(f"python ..\..\..\ort_trace.py --input {filename} -v --csv")     
    else:
        print("Error: Directory does not exist")
        

def model_dir_reader(dir_name, main_dir):
    # Read models from a directory where, ideally, .onnx files are stored. 
    # Runs ortperf.py, gets a trace, renames the trace to the AI/ML model name 
    # and stores it into a folder (create the folder) where it can then be processed. 


    if os.path.exists(dir_name) and os.path.isdir(dir_name):
        for model in os.listdir(dir_name):
            os.chdir(f'{main_dir}\{dir_name}\{model}')
            for filename in os.listdir():
                if fnmatch.fnmatch(filename, '*.onnx'):
                    print(f"Currently processing: {filename}...")
                    os.system(f"python ..\..\ortperf.py --model {filename}")
                    json_name = fnmatch.filter(os.listdir(os.getcwd()),'*.json')
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
                        os.system(f"python ..\..\..\ort_trace.py --input {sub_filename} -v --csv")   
                    print(f"{filename} processing: DONE")

                    # return "model_traces", main_dir

    elif os.path.exists(dir_name) and os.path.isfile(dir_name):
        if fnmatch.fnmatch(dir_name, '*.onnx'):
            print(f"Currently processing: {filename}...")
            os.system(f"python {main_dir}\ortperf.py --model {filename}")
            json_name = fnmatch.filter(os.listdir(),'*.json')
            os.rename(json_name[0], f'{model}_trace.json')
            if not os.path.exists('model_traces'):
                os.mkdir('model_traces')
            shutil.move(f'{model}_trace.json', 'model_traces')
            print(f"{filename} processing: DONE")

            # return "model_traces", main_dir

def get_args():
    '''
    Arg list:
        --dir [directory path(folder name if trace_parser is in the same directory)]
    '''   
    parser = argparse.ArgumentParser(description='Look at the code for flag descriptions')
    parser.add_argument("-d","--dir", required=True, help='directory path or folder name where model.onnx(s) are located')
    args = parser.parse_args()
    return args



def main():
    args = get_args()
    # results = model_dir_reader(args.dir, os.getcwd())
    model_dir_reader(args.dir, os.getcwd())
    # trace_dir_reader(results[0], results[1])
    return

if __name__ == "__main__":
    main()