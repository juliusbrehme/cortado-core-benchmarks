import sys
from cortado_core.visual_query_language.benchmark.query_miner import QueryMiner
from cortado_core.visual_query_language.benchmark.experiment import Experiment, BenchmarkExecutor
from cortado_core.visual_query_language.benchmark.utils import load_variants
from cortado_core.visual_query_language.query import QueryType


if __name__ == "__main__":
    for variants_name in ["bpi2012", "bpi2017", "bpi2019"]:
        print(f"> Loading {variants_name} variants...")
        variants = load_variants(variants_name)
        print(f"  Loaded {len(variants)} variants.")

        EXPERIMENT_QUERIES = 1000
        
        ### We will now compare the best algorithms
        PLOT_CONFIG = {
            "y_min": 1,
            "y_max": 5*10**2
        }

        # --- Experiment 2: Varying query length scaled
        print("> Experiment 2: Varying query length scaled...")
        experiment2 = Experiment(
            variants,
            QueryMiner(variants, cut_probability=1.0, disable_random_walk=True),
            num_queries=EXPERIMENT_QUERIES,
            query_types=[QueryType.VM, QueryType.VM_LAZY, QueryType.DFS, QueryType.BFS],
            desc="Varying query length",
            exp_id="query_length",
            plot_config=PLOT_CONFIG,
            results_dir=f"results/{variants_name}",
        ) 

        # --- Experiment 3: Varying parallelism
        print("> Experiment 3: Varying parallelism...")
        experiment3 = Experiment(
            variants,
            QueryMiner(variants, cut_probability=1.0, disable_random_walk=True),
            num_queries=EXPERIMENT_QUERIES,
            query_types=[QueryType.VM, QueryType.VM_LAZY, QueryType.DFS, QueryType.BFS],
            desc="Varying parallelism",
            exp_id="parallelism",
            plot_config=PLOT_CONFIG,
            results_dir=f"results/{variants_name}",
        )   

        PLOT_CONFIG = {
            "y_min": 1,
            "y_max": 10**4
        }

        # --- Experiment 4: Varying optionals
        print("> Experiment 4: Varying optionals...")
        experiment4 = Experiment(
            variants,
            QueryMiner(variants, cut_probability=0.2, mutation_probabilities={
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
        )

        # --- Experiment 5: Different anythings
        print("> Experiment 5: Different anythings...")
        experiment5 = Experiment(
            variants,
            QueryMiner(variants, cut_probability=0.2, mutation_probabilities={
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
        )

        # --- Experiment 6: Varying wildcards
        print("> Experiment 6: Varying wildcards...")
        experiment6 = Experiment(
            variants,
            QueryMiner(variants, cut_probability=0.2, mutation_probabilities={
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
        )

        experiment7 = Experiment(
            variants,
            QueryMiner(variants, cut_probability=0.2, mutation_probabilities={
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
        )

        # --- Run Experiments ---
        experiments = [experiment2, experiment3, experiment4, experiment5, experiment6, experiment7]
        
        # Use the shared executor
        executor = BenchmarkExecutor(experiments)
        executor.run()
        
        # Generate plots for each experiment
        for experiment in experiments:
            experiment.plot()