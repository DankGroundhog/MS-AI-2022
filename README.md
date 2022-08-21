# Summer 2022: AI Platform Internship Project

>**Project Name**: Project Fenix: Synthetic Model Creation for Research.
>By: Felix M. Perez Quiñones

>**Goal**: creating a set of tools that will allow the user to generate a "synthetic" AI/ML model that is optimized and generalized for benchmarking purposes. This will allow Microsoft customers to access a model benchmarking tool that works for any model and does not compromise the company's internal security.

The process on how this works is described as follows:

* Create a folder "`[name]`" where the model folders would be stored in base directory.
* Move model folders into `[name]` folder.
* Run ``trace_parser.py`` with parameter `--dir "[name]"`
* `trace_parser.py` will use `ortperf.py` first to acquire a trace of `X` model which will be located inside the model's own folder. Afterwards, it renames the trace to "`[model name]_trace`" so it is easier to identify. 
* ``trace_parser.py`` creates a folder inside `X` model directory and moves said renamed trace inside.
* ``trace_parser.py`` then runs `ort_trace.py` on the renamed trace to extract specific fields that will be used as data for the "synthetic" model later.
* After the trace is processed, CSV and JSON formatted files will be generated containing detailed information on key aspects needed for the synthetic model creation. Inluded in them is a calculator for ratios of the set of models, ``stats_calculator.py`` provided in the model folder. This can be used to do a ratio-based distribution of OPs in the synthetic model.
* ``model_maker.py`` is then used to export ``process_and_make`` which is called to automate the synthetic model creation process. It takes the verbose version of the outputs to ensure maximum amount of data is being utilized in the model creation. 
* ``synthetic_model.onnx`` is then created and placed inside the models folder for convenience.

**JUSTIFICATION**
* Can be released to external customers/users without compromising internal Microsoft security.
* Makes the benchmarking/testing process easier.
    * Run/optimize/benchmark 1 model instead of N models.
    * No issues modifying synthetic model rather than established ones.
* Optimizations to synthetic model could translate into optimizations to other models.

**RESOURCES**
* [ONNX](https://github.com/onnx/onnx)
* [ONNX Tutorials](https://github.com/onnx/tutorials)
* [ONNX Runtime API](https://onnxruntime.ai/docs/api/)
* [Loading an ONNX Model](https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md)
* [Model Operators - OPs](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
    * Points of interest:
        * onnx -> helper.py
        * onnx -> onnx.proto
        * [Graph help](https://github.com/onnx/tensorflow-onnx/blob/main/tf2onnx/graph.py)

Project is still WIP as of: 8/10/2022