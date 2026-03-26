import copy
import time
import signal
from typing import List, Tuple, Optional
from cortado_core.utils.split_graph import SequenceGroup
from cortado_core.visual_query_language.query import create_query_instance, QueryType


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Query execution timed out")


def run_matching(query, variants, query_type):
    instance = create_query_instance(query, query_type)
    counter = 0
    for v in variants:
        counter += instance.match(v)
    return counter


def run_benchmark(query, variants: List[SequenceGroup], query_types: List[QueryType], iterations: int = 100,
                  timeout_sec: Optional[int] = None) -> Tuple[List[float], int]:
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
        query_copy = copy.deepcopy(query)  # we need a fresh query copy for each algorithm (they may modify it)

        # Set the signal alarm if a timeout is provided
        if timeout_sec is not None:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_sec)

        start_time = time.perf_counter()
        try:
            for _ in range(iterations):
                run_matching(query_copy, variants, query_type)

            elapsed_time = time.perf_counter() - start_time
            avg_time = (elapsed_time / iterations) * 1000  # Convert to ms

        except TimeoutException:
            # Cap the time at the timeout limit for the plots
            avg_time = timeout_sec * 1000.0
            # We don't print here to avoid spamming the console during multiprocessing,
            # but you can add a print statement back if you want to track it.

        except Exception as e:
            print(f"[!] Error timing {query_type}: {e}")
            avg_time = float('inf')

        finally:
            # Always disable the alarm after the run
            if timeout_sec is not None:
                signal.alarm(0)

        timings.append(avg_time)

    return timings, match_count
