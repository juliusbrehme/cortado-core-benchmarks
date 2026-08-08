import sys
from cortado_core.visual_query_language.benchmark.query_miner import QueryMiner
from cortado_core.visual_query_language.benchmark.experiment import Experiment, BenchmarkExecutor
from cortado_core.visual_query_language.benchmark.utils import load_variants
from cortado_core.visual_query_language.query import QueryType


# When invoked with `--plot-only` (or `--plot`), we skip the (expensive) benchmark
# run entirely and only (re)generate the plots from the already existing CSVs in
# `results/<dataset>/<experiment>/results.csv`. This lets us iterate on plot styling
# without re-running the whole study.
PLOT_ONLY = any(arg in ("--plot-only", "--plot") for arg in sys.argv[1:])


# Plot style applied to ALL experiments: "box" (grouped boxplots) or "line"
# (median trend + interquartile band). Change the default here, or override on
# the command line with `--style line` / `--style=box`.
def _get_style(argv, default="box"):
    for i, arg in enumerate(argv):
        if arg == "--style" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--style="):
            return arg.split("=", 1)[1]
    return default


PLOT_STYLE = _get_style(sys.argv[1:])


def make_miner(variants, **kwargs):
    """Build a QueryMiner, or None in plot-only mode (no variants needed to plot)."""
    if PLOT_ONLY:
        return None
    return QueryMiner(variants, **kwargs)


if __name__ == "__main__":
    if PLOT_ONLY:
        print("> Plot-only mode: loading results from CSVs, skipping benchmark run.")

    for variants_name in ["bpi2012", "bpi2017", "bpi2019"]:
        if PLOT_ONLY:
            variants = None
        else:
            print(f"> Loading {variants_name} variants...")
            variants = load_variants(variants_name)
            print(f"  Loaded {len(variants)} variants.")

        EXPERIMENT_QUERIES = 1000

        TIMEOUT = 60

        SEED = 42

        # Publication defaults: render figures at their final on-page size (~half a
        # text width, so two fit across one page) with ~8pt text, so nothing gets
        # shrunk (and blurred) by LaTeX. Vector PDF/SVG output stays crisp.
        FONT = {'size': 8}
        FIGSIZE = (3.4, 2.7)
        
        ### We will now compare the best algorithms
        PLOT_CONFIG = {
            "y_min": 10**1,
            "y_max": 10**3,
            "x_max": 40,
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
            results_dir=f"results/{variants_name}",
            timeout_sec=TIMEOUT
        )

        PLOT_CONFIG = {
            "y_min": 10**1,
            "y_max": 10**3,
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
            results_dir=f"results/{variants_name}",
            timeout_sec=TIMEOUT
        )

        PLOT_CONFIG = {
            "y_min": 10**1,
            "y_max": 10**4,
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
            results_dir=f"results/{variants_name}",
            timeout_sec=TIMEOUT
        )

        PLOT_CONFIG = {
            "y_min": 10**1,
            "y_max": 10**4,
            "binned": False,
            "font": FONT,
            "x_max": 10,
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
            results_dir=f"results/{variants_name}",
            timeout_sec=TIMEOUT
        )

        PLOT_CONFIG = {
            "y_min": 10**1,
            "y_max": 10**4,
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
            results_dir=f"results/{variants_name}",
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
            results_dir=f"results/{variants_name}",
            timeout_sec=TIMEOUT
        )

        # --- Run Experiments ---
        experiments = [experiment2, experiment3, experiment4, experiment5, experiment6, experiment7]

        # Run the benchmark unless we only want to (re)generate plots from existing CSVs.
        if not PLOT_ONLY:
            executor = BenchmarkExecutor(experiments)
            executor.run()

        # Apply the global plot style (box/line) to every experiment.
        for experiment in experiments:
            experiment.plot_config["style"] = PLOT_STYLE

        # Generate plots for each experiment (reads results.csv from disk).
        for experiment in experiments:
            experiment.plot()
