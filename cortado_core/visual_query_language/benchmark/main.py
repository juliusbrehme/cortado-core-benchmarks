import sys
from pathlib import Path
from cortado_core.visual_query_language.benchmark.query_miner import QueryMiner
from cortado_core.visual_query_language.benchmark.experiment import Experiment, BenchmarkExecutor
from cortado_core.visual_query_language.benchmark.utils import load_variants
from cortado_core.visual_query_language.query import QueryType


PLOT_ONLY = any(arg in ("--plot-only", "--plot") for arg in sys.argv[1:])

def _get_style(argv, default="box"):
    for i, arg in enumerate(argv):
        if arg == "--style" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--style="):
            return arg.split("=", 1)[1]
    return default

PLOT_STYLE = _get_style(sys.argv[1:])


def _get_list_opt(argv, name):
    """Read a comma-separated (or repeated) option, e.g. `--only a,b` /
    `--only=a --only b`. Returns a set of values, or None if the option is absent."""
    vals = []
    for i, arg in enumerate(argv):
        if arg == name and i + 1 < len(argv):
            vals.extend(argv[i + 1].split(","))
        elif arg.startswith(name + "="):
            vals.extend(arg.split("=", 1)[1].split(","))
    cleaned = {v.strip() for v in vals if v.strip()}
    return cleaned or None


ONLY_DATASETS = _get_list_opt(sys.argv[1:], "--dataset")
ONLY_EXPERIMENTS = _get_list_opt(sys.argv[1:], "--only")


def make_miner(variants, **kwargs):
    """Build a QueryMiner, or None in plot-only mode (no variants needed to plot)."""
    if PLOT_ONLY:
        return None
    return QueryMiner(variants, **kwargs)


if __name__ == "__main__":
    if PLOT_ONLY:
        print("> Plot-only mode: loading results from CSVs, skipping benchmark run.")

    for variants_name in ["bpi2012", "bpi2017", "bpi2019"]:

        if ONLY_DATASETS is not None and variants_name not in ONLY_DATASETS:
            continue

        if PLOT_ONLY:
            variants = None
        else:
            print(f"> Loading {variants_name} variants...")
            variants = load_variants(variants_name)
            print(f"  Loaded {len(variants)} variants.")

        EXPERIMENT_QUERIES = 10000

        TIMEOUT = 60

        SEED = 42

        RESULTS_DIR = Path(__file__).parent / f"resources/results/{variants_name}"

        # Publication defaults: render figures at their final on-page size (~half a
        # text width, so two fit across one page) with ~8pt text, so nothing gets
        # shrunk (and blurred) by LaTeX. Vector PDF/SVG output stays crisp.
        FONT = {'size': 8}
        FIGSIZE = (3.4, 2.7)
        
        ### We will now compare the best algorithms
        PLOT_CONFIG = {
            "y_min": 10**1,
            "y_max": 10**5,
            "x_max": 100,
            "binned": False,
            "font": FONT,
            "figsize": FIGSIZE
        }

        # --- Experiment 2: Varying query length scaled
        print("> Experiment 2: Varying query length scaled...")
        experiment2 = Experiment(
            variants,
            make_miner(variants, seed=SEED, cut_probability=1.0, disable_random_walk=True),
            num_queries=EXPERIMENT_QUERIES,
            query_types=[QueryType.VM, QueryType.VM_LAZY, QueryType.DFS, QueryType.BFS],
            desc="Varying query length",
            exp_id="query_length",
            plot_config=PLOT_CONFIG,
            results_dir=RESULTS_DIR,
            timeout_sec=TIMEOUT
        )

        PLOT_CONFIG = {
            "y_min": 10**1,
            "y_max": 10**5,
            "x_max": 10,
            "binned": False,
            "font": FONT,
            "figsize": FIGSIZE
        }
        # --- Experiment 3: Varying parallelism
        print("> Experiment 3: Varying parallelism...")
        experiment3 = Experiment(
            variants,
            make_miner(variants, seed=SEED, cut_probability=1.0, disable_random_walk=True),
            num_queries=EXPERIMENT_QUERIES,
            query_types=[QueryType.VM, QueryType.VM_LAZY, QueryType.DFS, QueryType.BFS],
            desc="Varying parallelism",
            exp_id="parallelism",
            plot_config=PLOT_CONFIG,
            results_dir=RESULTS_DIR,
            timeout_sec=TIMEOUT
        )

        PLOT_CONFIG = {
            "y_min": 10**1,
            "y_max": 10**5,
            "binned": False,
            "x_max": 10,
            "font": FONT,
            "figsize": FIGSIZE
        }

        #--- Experiment 4: Varying optionals
        print("> Experiment 4: Varying optionals...")
        experiment4 = Experiment(
            variants,
            make_miner(variants, seed=SEED, cut_probability=0.2, mutation_probabilities={
                "make_wildcard": 0.0,
                "make_anything": 0.0,
                "make_optional": 0.95,
                "make_choice": 0.0,
                "make_no_match": 0.05,
            }),
            num_queries=EXPERIMENT_QUERIES,
            query_types=[QueryType.VM, QueryType.VM_LAZY, QueryType.DFS, QueryType.BFS],
            desc="Varying optionals",
            exp_id="optionals",
            plot_config=PLOT_CONFIG,
            results_dir=RESULTS_DIR,
            timeout_sec=TIMEOUT
        )

        PLOT_CONFIG = {
            "y_min": 10**1,
            "y_max": 10**5,
            "binned": False,
            "font": FONT,
            "x_max": 7,
            "figsize": FIGSIZE
        }

        # --- Experiment 5: Different anythings
        print("> Experiment 5: Different anythings...")
        experiment5 = Experiment(
            variants,
            make_miner(variants, seed=SEED, cut_probability=0.2, mutation_probabilities={
                "make_wildcard": 0.0,
                "make_anything": 0.95,
                "make_optional": 0.0,
                "make_choice": 0.0,
                "make_no_match": 0.05,
            }),
            num_queries=EXPERIMENT_QUERIES,
            query_types=[QueryType.VM, QueryType.VM_LAZY, QueryType.DFS, QueryType.BFS],
            desc="Varying anythings",
            exp_id="anythings",
            plot_config=PLOT_CONFIG,
            results_dir=RESULTS_DIR,
            timeout_sec=TIMEOUT
        )

        PLOT_CONFIG = {
            "y_min": 10**1,
            "y_max": 10**5,
            "binned": False,
            "font": FONT,
            "x_max": 10,
            "figsize": FIGSIZE
        }

        # --- Experiment 6: Varying wildcards
        print("> Experiment 6: Varying wildcards...")
        experiment6 = Experiment(
            variants,
            make_miner(variants, seed=SEED, cut_probability=0.2, mutation_probabilities={
                "make_wildcard": 0.95,
                "make_anything": 0.0,
                "make_optional": 0.0,
                "make_choice": 0.0,
                "make_no_match": 0.05,
            }),
            num_queries=EXPERIMENT_QUERIES,
            query_types=[QueryType.VM, QueryType.VM_LAZY, QueryType.DFS, QueryType.BFS],
            desc="Varying wildcards",
            exp_id="wildcards",
            plot_config=PLOT_CONFIG,
            results_dir=RESULTS_DIR,
            timeout_sec=TIMEOUT
        )

        experiment7 = Experiment(
            variants,
            make_miner(variants, seed=SEED, cut_probability=0.2, mutation_probabilities={
                "make_wildcard": 0.0,
                "make_anything": 0.0,
                "make_optional": 0.0,
                "make_choice": 0.95,
                "make_no_match": 0.05,
            }),
            num_queries=EXPERIMENT_QUERIES,
            query_types=[QueryType.VM, QueryType.VM_LAZY, QueryType.DFS, QueryType.BFS],
            desc="Varying choices",
            exp_id="choices",
            plot_config=PLOT_CONFIG,
            results_dir=RESULTS_DIR,
            timeout_sec=TIMEOUT
        )

        # --- Run Experiments ---
        experiments = [experiment2, experiment3, experiment4, experiment5, experiment6, experiment7]

        # Run the benchmark unless we only want to (re)generate plots from existing CSVs.
        # With --only, run just the selected experiments; the rest keep their CSVs.
        if not PLOT_ONLY:
            to_run = experiments
            if ONLY_EXPERIMENTS is not None:
                to_run = [e for e in experiments if e.id in ONLY_EXPERIMENTS]
            if to_run:
                print(f"> Running experiments: {[e.id for e in to_run]}")
                BenchmarkExecutor(to_run).run()
            else:
                print("> No experiments selected to run (check --only values).")

        # Apply the global plot style (box/line) to every experiment.
        for experiment in experiments:
            experiment.plot_config["style"] = PLOT_STYLE

        # Generate plots for each experiment (reads results.csv from disk).
        for experiment in experiments:
            experiment.plot()
