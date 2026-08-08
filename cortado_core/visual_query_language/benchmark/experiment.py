import copy
import multiprocessing
import os
from typing import Optional, List, Dict, Any, Tuple
import uuid
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


# --- Publication styling ---------------------------------------------------
# Colorblind-safe categorical palette (Okabe-Ito), assigned to algorithms in a
# FIXED order so a given algorithm keeps the exact same color in every figure of
# the paper. Never let seaborn cycle these per-figure.
ALGO_COLORS = {
    "VM": "#0072B2",       # blue
    "VM_LAZY": "#E69F00",  # orange
    "DFS": "#009E73",      # bluish green
    "BFS": "#CC79A7",      # reddish purple (kept well apart from the orange)
}
# Extra hues used only if an experiment adds algorithms beyond the four above.
_EXTRA_COLORS = ["#D55E00", "#56B4E9", "#F0E442", "#000000"]

# Short, print-friendly legend labels.
ALGO_LABELS = {
    "VM": "VM",
    "VM_LAZY": "VM (lazy)",
    "DFS": "DFS",
    "BFS": "BFS",
}


def _column_major(items, ncol):
    """Reorder items so a row-major legend with ``ncol`` columns fills up
    column-by-column. With 4 items and ncol=2 the columns become [0,1] and
    [2,3], i.e. the 3rd/4th entries sit *under* the 1st/2nd (BFS under DFS)."""
    n = len(items)
    nrows = -(-n // ncol)  # ceil division
    ordered = []
    for r in range(nrows):
        for c in range(ncol):
            idx = c * nrows + r
            if idx < n:
                ordered.append(items[idx])
    return ordered


def algo_palette(algorithms):
    """Return a stable {algorithm: color} map for the given algorithm names."""
    palette = {}
    spare = iter(_EXTRA_COLORS)
    for algo in algorithms:
        palette[algo] = ALGO_COLORS.get(algo) or next(spare, "#666666")
    return palette


def set_publication_style(font_size=8):
    """Global matplotlib rcParams tuned for small, print-quality figures.

    Small figures live or die on typography and restraint: modest fonts, thin
    lines, a recessive grid and no heavy frame. We set this once per plot() call.
    """
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.4,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.2,
        "savefig.dpi": 300,
        "figure.dpi": 150,
        # Keep text as text in vector output (so LaTeX/Illustrator can select it).
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    # Trim the top/right spines for a lighter look.
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


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
        algorithms = [qt.name for qt in self.query_types]

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

        # --- Resolve styling config (all optional, sensible paper defaults) ---
        cfg = self.plot_config
        figsize = cfg.get("figsize") or (3.4, 2.7)
        font_size = (cfg.get("font") or {}).get("size", 8)
        style = cfg.get("style", "box")          # "box" or "line" (median + IQR band)
        show_title = cfg.get("show_title", False)  # paper: caption carries the title
        show_fliers = cfg.get("showfliers", False)  # hide outlier dots -> cleaner small plots
        legend_pos = cfg.get("legend", "top")      # "top", "best", or False
        max_xticks = cfg.get("max_xticks", 12)     # thin x labels beyond this many groups

        set_publication_style(font_size)
        palette = algo_palette(available_algos)
        pretty = lambda a: ALGO_LABELS.get(a, a)

        def finalize(ax, col):
            """Shared axis cosmetics: log y, labels, limits, timeout line, legend."""
            ax.set_yscale("log")
            ax.set_ylabel("Runtime (ms)")
            ax.set_xlabel(col_descriptions[col])
            if show_title:
                ax.set_title(f"{col_descriptions[col]} ({self.desc})")
            ax.grid(True, which="major", ls="--", alpha=0.4)
            ax.margins(x=0.02)

            if "y_min" in cfg:
                ax.set_ylim(bottom=cfg["y_min"])
            if "y_max" in cfg:
                ax.set_ylim(top=cfg["y_max"])

            algo_handles = [Line2D([0], [0], color=palette[a], lw=2.2, label=pretty(a))
                            for a in available_algos]

            # Two columns, filled column-major, so e.g. BFS lines up under DFS.
            ncol = 2 if len(algo_handles) >= 3 else 1
            handles = _column_major(algo_handles, ncol)

            if self.timeout_sec is not None:
                timeout_ms = self.timeout_sec * 1000
                ax.axhline(y=timeout_ms, color="0.35", linestyle=":", linewidth=1.0)
                # Appended last, so it sits on its own row beneath the grid.
                handles.append(Line2D([0], [0], color="0.35", linestyle=":", lw=1.0,
                                      label=f"Timeout ({self.timeout_sec}s)"))

            # Drop any auto-legend seaborn added; place our own clean one.
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            if legend_pos == "top":
                ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                          ncol=ncol, frameon=False, handlelength=1.4,
                          columnspacing=1.0, handletextpad=0.4, borderaxespad=0.1)
            elif legend_pos:
                ax.legend(handles=handles, loc=legend_pos, ncol=ncol, frameon=False,
                          handlelength=1.4, handletextpad=0.4)

        def save_current(fig, name):
            pdf.savefig(fig, bbox_inches="tight")
            for ext in ("pdf", "png", "svg"):
                fig.savefig(os.path.join(self.results_dir, f"{name}.{ext}"),
                            bbox_inches="tight")
            plt.close(fig)

        with PdfPages(self.pdf_file) as pdf:

            # --- Plot A: Runtime vs each varied metric ---
            for col in active_cols:
                # Focus on the region of interest: cap the metric at x_max (if set) so
                # small figures don't try to cram dozens of x-groups across a few inches.
                plot_df = df_melted.copy()
                if "x_min" in cfg:
                    plot_df = plot_df[plot_df[col] >= cfg["x_min"]]
                if "x_max" in cfg:
                    plot_df = plot_df[plot_df[col] <= cfg["x_max"]]
                if plot_df.empty:
                    continue

                # A boxplot with hundreds of integer x-values is unreadable at this
                # size; fall back to the compact line style for high-cardinality metrics.
                n_unique = plot_df[col].nunique()
                effective_style = style
                if style == "box" and n_unique > 2 * max_xticks:
                    effective_style = "line"

                fig, ax = plt.subplots(figsize=figsize)

                if effective_style == "line":
                    # Median trend + interquartile band per algorithm: reads cleanly
                    # even at half-page width and shows the scaling behaviour directly.
                    for algo in available_algos:
                        sub = plot_df[plot_df["Algorithm"] == algo]
                        grp = sub.groupby(col)["Runtime"]
                        stats = grp.agg(["median"])
                        q1 = grp.quantile(0.25)
                        q3 = grp.quantile(0.75)
                        x = stats.index.values
                        ax.fill_between(x, q1.values, q3.values, color=palette[algo],
                                        alpha=0.15, linewidth=0)
                        ax.plot(x, stats["median"].values, color=palette[algo],
                                marker="o", markersize=2.5, linewidth=1.3)
                else:
                    # Box style: thin lines, no flier clutter, fixed colorblind palette.
                    sns.boxplot(
                        data=plot_df, x=col, y="Runtime", hue="Algorithm",
                        hue_order=available_algos, palette=palette, ax=ax,
                        linewidth=0.5, fliersize=1.2, showfliers=show_fliers,
                        width=0.8,
                    )
                    # With many integer x-values, keep only every k-th tick label.
                    n_groups = plot_df[col].nunique()
                    if n_groups > max_xticks:
                        step = int(np.ceil(n_groups / max_xticks))
                        for i, lbl in enumerate(ax.get_xticklabels()):
                            lbl.set_visible(i % step == 0)

                finalize(ax, col)
                fig.tight_layout()
                save_current(fig, f"plot_{col}_vs_runtime")

            # --- Plot B: Overall runtime distribution per algorithm ---
            fig, ax = plt.subplots(figsize=figsize)
            sns.boxplot(data=df_melted, x="Algorithm", y="Runtime", ax=ax,
                        order=available_algos, palette=palette,
                        linewidth=0.5, fliersize=1.2, showfliers=show_fliers, width=0.7)
            ax.set_xticklabels([pretty(a) for a in available_algos])
            ax.set_yscale("log")
            ax.set_ylabel("Runtime (ms)")
            ax.set_xlabel("")
            if show_title:
                ax.set_title(f"Overall runtime ({self.desc})")
            ax.grid(True, which="major", ls="--", alpha=0.4)
            if "y_min" in cfg:
                ax.set_ylim(bottom=cfg["y_min"])
            if "y_max" in cfg:
                ax.set_ylim(top=cfg["y_max"])
            fig.tight_layout()
            save_current(fig, "plot_overall_runtime")

            # --- Plot C: Correlation Matrix (diagnostic) ---
            if len(active_cols) > 1:
                corr_cols = active_cols + available_algos
                corr_matrix = df[corr_cols].corr()

                fig, ax = plt.subplots(figsize=(max(figsize[0], 5), max(figsize[0], 5)))
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f",
                            ax=ax, annot_kws={"size": font_size - 2}, cbar=False,
                            vmin=-1, vmax=1)
                if show_title:
                    ax.set_title(f"Correlation ({self.desc})")
                fig.tight_layout()
                save_current(fig, "plot_correlation")

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
            ] + [qt.name for qt in exp.query_types]
            
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

