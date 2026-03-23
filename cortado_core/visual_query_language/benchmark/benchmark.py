from typing import List, Tuple
import copy
import timeit
from cortado_core.utils.split_graph import SequenceGroup
from cortado_core.visual_query_language.query import create_query_instance, QueryType


def run_matching(query, variants, query_type):
    instance = create_query_instance(query, query_type)
    counter = 0
    for v in variants:
        counter += instance.match(v)
    return counter

def run_benchmark(query, variants: List[SequenceGroup], query_types: List[QueryType], iterations: int = 100) -> Tuple[List[float], int]:
    """
    Runs benchmarks and counts matches.
    Returns: (List of runtimes in ms, Number of matched variants)
    """
    timings = []
    
    # 1. Get Match Count (We run once with VM to get the ground truth count)
    # We assume all algorithms return the same result (they should!)
    vm_instance = create_query_instance(query, QueryType.VM)
    match_count = sum(vm_instance.match(v) for v in variants)

    # 2. Run Timings
    for query_type in query_types:
        query_copy = copy.deepcopy(query) # we need a fresh query copy for each algorithm (they may modify it)
        timer = timeit.Timer(
            stmt=lambda: run_matching(query_copy, variants, query_type)
        )
        try:
            total_time = timer.timeit(number=iterations)
        except Exception as e:
            print(f"Error timing {query_type}: {e}")
            total_time = float('inf')
        avg_time = (total_time / iterations) * 1000  # Convert to ms
        timings.append(avg_time)
        
    return timings, match_count