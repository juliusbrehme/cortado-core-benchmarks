# Visual Query Language Benchmark for Cortado-Core
This repository contains benchmark resources and scripts for evaluating the performance of the Visual Query Language (POVQL) implementation in Cortado Core.

## Requirements
* Install Python 3.10.x (https://www.python.org/downloads/). Make sure to install a 64-BIT version.
* ~2 GB of free disk space (for the decompressed datasets)

## Setup

**1. Clone the repository**
 
```bash
git clone https://github.com/juliusbrehme/cortado-core-benchmarks.git
cd cortado-core-benchmarks
```

**2. Create and activate a virtual environment** (recommended)
 
macOS / Linux:
 
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```
 
Windows (PowerShell):
 
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**3. Install the dependencies**
 
```bash
pip install -r requirements.txt
```

**4. Decompress the datasets**
 
The datasets are stored as `.tar.gz` archives to keep the repository small. Extract them **in
place**, into `cortado_core/visual_query_language/benchmark/resources/`:
 
macOS / Linux:
 
```bash
cd cortado_core/visual_query_language/benchmark/resources
for f in bpi2012 bpi2017 bpi2019; do tar -xzf "$f.tar.gz"; done
cd -
```
 
Windows (PowerShell):
 
```powershell
cd cortado_core\visual_query_language\benchmark\resources
foreach ($f in "bpi2012","bpi2017","bpi2019") { tar -xzf "$f.tar.gz" }
cd ..\..\..\..
```
 
Afterwards the `resources` directory must contain `bpi2012.p`, `bpi2017.p` and `bpi2019.p`.
Verify with:
 
```bash
ls cortado_core/visual_query_language/benchmark/resources/*.p
```


## Running Benchmarks
_NOTE: Executing all benchmarks is compute heavy. If you do not want to reexecute everything from scratch, you can use the plot only option. This will load our results and plot them with your settings._

To run the benchmarks execute the main python script.
```python
python -m cortado_core.visual_query_language.benchmark.main
```

If you have already run the benchmark you can also make use of:
```python
python -m cortado_core.visual_query_language.benchmark.main --plot-only
```

By default, the script creates box plots, the figures from the paper (lineplots) are created with **(Recommended)**:
```python
python -m cortado_core.visual_query_language.benchmark.main --plot-only --style line
```

It is also possible to rerun specific experiments:
```python
python -m cortado_core.visual_query_language.benchmark.main  --dataset bpi2019
```

The script will load the benchmark datasets from the `/cortado_core/visual_query_language/benchmark/resources` subdirectory, automatically generate queries, measure execution times, and output the results to `/cortado_core/visual_query_language/benchmark/resources/results/<dataset>/<experiment>`.

### Options for the benchmarks
It is possible to set a timeout for the benchmarks. To set a timeout, set the parameter timeout_sec with a number representing seconds (Defaults to 60-seconds). 
For a 2-second timeout:
```
Experiment(
    variants,
    QueryMiner(variants, cut_probability=1.0, disable_random_walk=True),
    num_queries=EXPERIMENT_QUERIES,
    query_types=[QueryType.VM, QueryType.VM_LAZY, QueryType.DFS, QueryType.BFS],
    desc="Varying query length",
    exp_id="query_length",
    plot_config=PLOT_CONFIG,
    results_dir=f"results/{variants_name}",
    timeout_sec=2
)
```

## Resources
 
`cortado_core/visual_query_language/benchmark/resources/` holds the benchmark datasets and the
generated queries and results. The plots used in the paper are committed under
[`resources/results`](cortado_core/visual_query_language/benchmark/resources/results).

