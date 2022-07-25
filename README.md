# Summer 2022: AI Platform Internship Project

>**Project Name**: Synthetic Model Creation for Optimization/Benchmarking Research.

>**GOAL**: creating a set of tools that will allow the user to generate a "synthetic" AI/ML model that is optimized and generalized for benchmarking purposes. This will allow Microsoft customers to access a model benchmarking tool that works for any model and does not compromise the company's internal security.

The process on how is described as follows:

* User creates a folder "`name`" where the model folders would be stored in base directory.
* Move any amount of model folders into `name` folder.
* Run ``trace_parser.py`` with parameter `--dir "[name]"`
* `trace_parser.py` will use `ortperf.py` first to acquire a trace of `X` model which will be located inside the model's folder. Afterwards, it renames the trace to "`[model name]_trace`" so it is easier to identify. 
* ``trace_parser.py`` creates a folder inside `X` model directory and moves said renamed trace inside.
* ``trace_parser.py`` then runs `ort_trace.py` on the renamed trace to extract specific fields that will be used as data for the "synthetic" model later.
* After the trace is processed, CSV and JSON formatted files will be generated containing detailed information on key aspects needed for the synthetic model creation. Inluded in them is a calculator for ratios of the set of models, ``stats_calculator.py`` provided in the model folder. This can be used to do a ratio-based distribution of OPs in the synthetic model.
* ``model_maker.py`` is then used to export ``process_and_make`` which is called to automate the synthetic model creation process. It takes the verbose version of the outputs to ensure maximum amount of data is being utilized in the model creation. 
* ``synthetic_model.onnx`` is then created and placed inside the models folder for convenience.

**JUSTIFICATION???**