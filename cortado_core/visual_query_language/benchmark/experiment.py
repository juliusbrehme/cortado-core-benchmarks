import copy
import multiprocessing
import os
from typing import Optional, List, Dict, Any, Tuple
import uuid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
import io
from matplotlib.backends.backend_pdf import PdfPages
from cortado_core.visual_query_language.benchmark.benchmark import run_benchmark
from cortado_core.visual_query_language.benchmark.query_miner import QueryMiner
from cortado_core.visual_query_language.benchmark.utils import check_double_anything, serialize_group
from cortado_core.visual_query_language.query import QueryType


# --- Global Scope for Workers ---
# These must remain global/module-level so they can be initialized in worker processes
global_variants = None
global_miner = None

def init_worker(variants_data):
    """
    Initializes the worker process with the dataset.
    This runs once per process, not per query.
    """
    global global_variants # pylint: disable=global-statement
    global global_miner # pylint: disable=global-statement
    global_variants = variants_data
    # We create a default miner here strictly for computing complexity metrics inside the worker
    global_miner = QueryMiner(global_variants)


def process_query(args) -> Optional[Tuple[dict, int]]:
    """
    args: (query, query_types, exp_idx, timeout_sec)
    """
    query, query_types, exp_idx, timeout_sec = args

    if check_double_anything(query):
        return None

    complexity = global_miner.compute_query_complexity(query)
    query_copy = copy.deepcopy(query)

    # Pass timeout_sec down (it can be an int or None)
    timings, match_count = run_benchmark(
        query_copy, global_variants, query_types, 1, timeout_sec=timeout_sec
    )

    result = {
        "num_elements": complexity["elements"],
        "num_wildcards": complexity["wildcards"],
        "num_anything": complexity["anythings"],
        "num_optionals": complexity["optionals"],
        "num_parallels": complexity["parallels"],
        "num_choices": complexity["choices"],
        "tree_depth": complexity["depth"],
        "matches": match_count,
    }

    for qt, timing in zip(query_types, timings):
        result[qt.name.lower()] = timing

    return result, exp_idx

class Experiment:
    def __init__(self, 
                 variants, 
                 miner: QueryMiner, 
                 num_queries: int, 
                 query_types: List[QueryType],
                 desc: str = "", 
                 exp_id: str = None,
                 plot_config: Dict[str, Any] = None,
                 results_dir: str = None,
                 timeout_sec: Optional[int] = None):
        """
        :param variants: The event log/graph variants (passed to workers)
        :param miner: The configured QueryMiner instance (used for generation)
        :param num_queries: How many queries to generate
        :param query_types: List of QueryTypes to benchmark
        :param desc: Human readable description
        :param exp_id: UUID for file naming
        :param plot_config: Configuration for plotting (e.g. scales)
        """
        self.variants = variants
        self.miner = miner
        self.num_queries = num_queries
        self.query_types = query_types
        self.desc = desc
        self.id = exp_id if exp_id else str(uuid.uuid4())
        self.queries = []
        self.plot_config = plot_config or {}
        
        # Setup results directory
        self.results_dir =os.path.join(results_dir or "results", self.id)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.csv_file = os.path.join(self.results_dir, "results.csv")
        self.pdf_file = os.path.join(self.results_dir, "plots.pdf")
        self.timeout_sec = timeout_sec

    def generate(self):
        print(f"[{self.desc}] Generating {self.num_queries} queries...")
        self.queries = self.miner.generate_queries(self.num_queries)
        
        # Save queries to text file
        with open(os.path.join(self.results_dir, "queries.txt"), "w", encoding="utf-8") as f:
            s = io.StringIO()
            for query in self.queries:
                serialize_group(s, query)
                s.write("\n")
            f.write(s.getvalue())
    
    def run(self):
        # Convenience method to run just this experiment
        executor = BenchmarkExecutor([self])
        executor.run()

    def plot(self):
        print(f"[{self.desc}] generating plots -> {self.results_dir}")

        try:
            df = pd.read_csv(self.csv_file)
        except FileNotFoundError:
            print(f"Error: {self.csv_file} not found. Run .run() first.")
            return

        if df.empty:
            print("Error: DataFrame is empty.")
            return

        # 1. Determine which complexity columns actually have data (variance > 0)
        potential_cols = [
            "num_elements", "num_wildcards", "num_anything", "num_optionals", "num_parallels", "num_choices",
            "tree_depth", "matches"
        ]

        col_descriptions = {
            "num_elements": "Number of Nodes",
            "num_wildcards": "Number of Wildcards",
            "num_anything": "Number of Anythings",
            "num_optionals": "Number of Optionals",
            "num_parallels": "Number of Parallels",
            "num_choices": "Number of Choices",
            "tree_depth": "Tree Depth",
            "matches": "Number of Matches",
        }

        # Filter: keep column only if max > min (i.e., not constant)
        active_cols = [col for col in potential_cols if df[col].max() > df[col].min()]

        # Prepare for plotting
        algorithms = [qt.name.lower() for qt in self.query_types]

        # Check if algorithms are in df columns
        available_algos = [algo for algo in algorithms if algo in df.columns]
        if not available_algos:
            print("No algorithm columns found in results.")
            return

        df_melted = df.melt(
            id_vars=active_cols,  # Only keep active cols as identifiers
            value_vars=available_algos,
            var_name="Algorithm",
            value_name="Runtime"
        )

        with PdfPages(self.pdf_file) as pdf:

            # --- Plot A: Runtime Distribution vs Metrics ---
            for col in active_cols:
                fig, ax = plt.subplots(figsize=(10, 6))

                # 1. Cut off the long tail of extreme outliers to focus on the important data.
                # We use the 95th percentile (keeps the bottom 95% of data).
                cutoff_val = df[col].quantile(1)

                # Fallback: If your data is highly skewed (e.g., 95% of queries have 0 optionals),
                # the cutoff might equal the minimum. In that case, use the absolute max.
                if cutoff_val <= df[col].min():
                    cutoff_val = df[col].max()

                # Filter the melted dataframe for this specific plot
                plot_df = df_melted[df_melted[col] <= cutoff_val].copy()

                # 2. Determine how many unique values exist in this filtered column
                unique_vals = plot_df[col].nunique()

                plot_x_col = col
                if unique_vals > 60:
                    plot_x_col = f"{col}_binned"

                    # Create clean integer bin edges from the min to our cutoff
                    min_val = int(np.floor(plot_df[col].min()))
                    max_val = int(np.ceil(plot_df[col].max()))

                    num_bins = 12  # 12 bins usually fit perfectly on a 10x6 chart
                    bin_edges = np.linspace(min_val, max_val, num_bins + 1)

                    # Generate nice human-readable labels like "0-10", "11-20"
                    clean_labels = []
                    for i in range(num_bins):
                        left = int(np.round(bin_edges[i]))
                        right = int(np.round(bin_edges[i + 1]))

                        # Prevent overlapping labels (e.g., "0-10", "11-20" instead of "0-10", "10-20")
                        if i > 0:
                            left += 1

                            # If a bin is so small that left and right are the same (e.g., "5-5"), just write "5"
                        if left >= right:
                            clean_labels.append(f"{right}")
                        else:
                            clean_labels.append(f"{left}-{right}")

                    # Apply our custom bins and labels
                    plot_df[plot_x_col] = pd.cut(
                        plot_df[col],
                        bins=bin_edges,
                        labels=clean_labels,
                        include_lowest=True
                    )

                # IMPORTANT: We use plot_df now, not df_melted!
                sns.boxplot(
                    data=plot_df,
                    x=plot_x_col,
                    y='Runtime',
                    hue='Algorithm',
                    ax=ax,
                    fliersize=3,
                    linewidth=1
                )

                # Rotate labels so our new clean strings don't overlap
                if plot_x_col != col:
                    ax.tick_params(axis='x', rotation=45)

                if self.timeout_sec is not None:
                    timeout_ms = self.timeout_sec * 1000
                    ax.axhline(y=timeout_ms, color='red', linestyle='--', linewidth=1.5,
                               label=f'Timeout ({self.timeout_sec}s)')
                    ax.legend()

                ax.set_yscale('log')
                ax.set_title(f"Runtime Distribution vs {col_descriptions[col]} ({self.desc})")
                ax.set_ylabel("Runtime (ms) - Log Scale")
                ax.grid(True, which="major", ls="--", alpha=0.5)

                if 'y_min' in self.plot_config:
                    ax.set_ylim(bottom=self.plot_config['y_min'])
                if 'y_max' in self.plot_config:
                    ax.set_ylim(top=self.plot_config['y_max'])

                plt.tight_layout()
                pdf.savefig(fig)

                plot_filename = f"plot_{col}_vs_runtime"
                plt.savefig(os.path.join(self.results_dir, f"{plot_filename}.png"), bbox_inches='tight')
                plt.savefig(os.path.join(self.results_dir, f"{plot_filename}.svg"), bbox_inches='tight')

                plt.close(fig)

            # --- Plot B: Average Runtime over whole Experiment (Boxplot) ---
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df_melted, x='Algorithm', y='Runtime', ax=ax)
            ax.set_title(f"Runtime Distribution over Experiment ({self.desc})")
            ax.set_ylabel("Runtime (ms) - Log Scale")
            ax.set_yscale('log')
            ax.grid(True, which="major", ls="--", alpha=0.5)

            # Apply scaling if configured
            if 'y_min' in self.plot_config:
                ax.set_ylim(bottom=self.plot_config['y_min'])
            if 'y_max' in self.plot_config:
                ax.set_ylim(top=self.plot_config['y_max'])

            plt.tight_layout()
            pdf.savefig(fig)
            plt.savefig(os.path.join(self.results_dir, "plot_overall_runtime.png"))
            plt.savefig(os.path.join(self.results_dir, "plot_overall_runtime.svg"))
            plt.close(fig)

            # --- Plot C: Correlation Matrix ---
            # (Only if we have at least 2 columns to correlate)
            if len(active_cols) > 1:
                corr_cols = active_cols + available_algos
                corr_matrix = df[corr_cols].corr()

                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                ax.set_title(f'Correlation Matrix ({self.desc})')

                plt.tight_layout()
                pdf.savefig(fig)
                plt.savefig(os.path.join(self.results_dir, "plot_correlation.png"))
                plt.savefig(os.path.join(self.results_dir, "plot_correlation.svg"))
                plt.close(fig)

        print(f"[{self.desc}] Plots saved.")


class BenchmarkExecutor:
    def __init__(self, experiments: List[Experiment]):
        self.experiments = experiments
        
    def run(self):
        if not self.experiments:
            return

        # 1. Validate shared variants
        first_variants = self.experiments[0].variants
        for exp in self.experiments[1:]:
            if exp.variants is not first_variants:
                 # In a real scenario we might support this, but for now we warn or error
                 # The init_worker sets a global variable, so we can only support one variant set per pool.
                 raise ValueError("All experiments in one Executor run must share the same variants object.")

        # 2. Generate queries if needed and prepare CSVs
        total_tasks = 0
        tasks = [] # List of (query, query_types, exp_index)
        
        for i, exp in enumerate(self.experiments):
            if not exp.queries:
                exp.generate()
            
            # Initialize CSV
            columns = [
                "num_elements", "num_wildcards", "num_anything", "num_optionals", "num_parallels", "num_choices",
                "tree_depth", "matches"
            ] + [qt.name.lower() for qt in exp.query_types]
            
            pd.DataFrame(columns=columns).to_csv(exp.csv_file, index=False)
            
            # Prepare tasks
            for q in exp.queries:
                tasks.append((q, exp.query_types, i, exp.timeout_sec))
                
            total_tasks += len(exp.queries)

        # 3. Run Benchmark
        num_processes = max(1, multiprocessing.cpu_count() - 1)
        print(f"Starting benchmark for {len(self.experiments)} experiments with {num_processes} processes.")
        
        # We need to wrap the task to include the experiment index so we know where to save the result
        # But process_query is clean. Let's make a wrapper or just handle it.
        # We can't easily pass the exp_index to process_query and get it back if process_query returns a dict.
        # We can wrap the query in a tuple: ((query, query_types), exp_index)
        # But process_query expects just args.
        # Let's define a helper here or make process_query handle it?
        # Better: process_query returns the result dict. We can't attach metadata easily unless we modify process_query.
        # Let's modify process_query to return (result, input_args) or similar? 
        # No, imap_unordered returns results in arbitrary order. We need to know which experiment a result belongs to.
        # So we MUST pass the experiment ID/index into the worker and return it.
        
        # Redefining tasks to be: ((query, query_types, exp_idx))
        # And we need a wrapper function that calls process_query and adds exp_idx to result.
        
        with multiprocessing.Pool(processes=num_processes, initializer=init_worker, initargs=(first_variants,)) as pool:
            
            # Using imap_unordered for better performance
            iterator = pool.imap_unordered(process_query, tasks)
            
            # Buffers for each experiment
            results_buffers = {i: [] for i in range(len(self.experiments))}
            batch_save_size = 100
            
            # Progress bar
            for result_tuple in tqdm.tqdm(iterator, total=total_tasks, desc=f"Benchmarking {len(self.experiments)} experiments"):
                if result_tuple is None: 
                    continue
                
                result, exp_idx = result_tuple
                results_buffers[exp_idx].append(result)
                
                # Batch save for this experiment
                if len(results_buffers[exp_idx]) >= batch_save_size:
                    pd.DataFrame(results_buffers[exp_idx]).to_csv(self.experiments[exp_idx].csv_file, mode='a', header=False, index=False)
                    results_buffers[exp_idx] = []

            # Final flush for all experiments
            for i, buffer in results_buffers.items():
                if buffer:
                    pd.DataFrame(buffer).to_csv(self.experiments[i].csv_file, mode='a', header=False, index=False)

        print("All benchmarks finished.")

