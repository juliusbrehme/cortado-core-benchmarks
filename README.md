# Visual Query Language Benchmark for Cortado-Core
This directory contains benchmark resources and scripts for evaluating the performance of the Visual Query Language (POVQL) implementation in Cortado Core.

## Setup
* Install Python 3.10.x (https://www.python.org/downloads/). Make sure to install a 64-BIT version.
* Optional (recommended): Create a virtual environment (https://docs.python.org/3/library/venv.html) and activate it
* Install all packages required by cortado-core
  * Execute `pip install -r requirements.txt`


## Running Benchmarks
To run the benchmarks execute the main python script.
```python
python ./cortado_core/visual_query_language/benchmark/main.py
```
Right now for each Experiment there is a timelimit of 2sec set. It is possible to remove as well.

The script will load the benchmark datasets from the `/cortado_core/visual_query_language/benchmark/resources` subdirectory, automatically generate queries, measure execution times, and output the results in the `/cortado_core/visual_query_language/benchmark/resources/results/` subdirectory per dataset.

## Resource
The `/cortado_core/visual_query_language/benchmark/resources` subdirectory includes various benchmark datasets and query definitions used for testing and performance evaluation.
To decrease repository size the files are gzipped. Please decompress them before use.

## Plots
The plots used in the paper can be found [plots](./cortado_core/visual_query_language/benchmark/plots/)

