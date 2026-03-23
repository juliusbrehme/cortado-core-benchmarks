# Visual Query Language Benchmark
This directory contains benchmark resources and scripts for evaluating the performance of the Visual Query Language (VQL) implementation in Cortado Core.

## Running Benchmarks
To run the benchmarks execute the main python script.
```python
python main.py
```

The script will load the benchmark datasets from the `resources` subdirectory, automatically generate queries, measure execution times, and output the results in the `results/` subdirectory per dataset.

## Resource
The `resources` subdirectory includes various benchmark datasets and query definitions used for testing and performance evaluation.
To decrease repository size the files are gzipped. Please decompress them before use.