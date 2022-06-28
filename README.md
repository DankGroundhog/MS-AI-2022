# Summer 2022: AI Platform Internship Project

>By: Felix M. Perez Quiñones

>**Project Name**: Synthetic Model Creation for Optimization/Benchmarking Research.

>**Goal**: creating a set of tools that will allow the user to generate a "synthetic" AI/ML model that is optimized and generalized for benchmarking purposes. This will allow Microsoft customers to access a model benchmarking tool that works for any model and does not compromise the company's internal security.

The process on how this works is described as follows:

* Create a folder "`[name]`" where the model folders would be stored in base directory.
* Move model folders into `[name]` folder.
* Run ``trace_parser.py`` with parameter `--dir "[name]"`
* `trace_parser.py` will use `ortperf.py` first to acquire a trace of `X` model which will be located inside the model's own folder. Afterwards, it renames the trace to "`[model name]_trace`" so it is easier to identify. 
* ``trace_parser.py`` creates a folder inside `X` model directory and moves said renamed trace inside.
* ``trace_parser.py`` then runs `ort_trace.py` on the renamed trace to extract specific fields that will be used as data for the "synthetic" model later.
* **TO-DO : Finish this section.**

**ADD JUSTIFICATION MAYBE?**