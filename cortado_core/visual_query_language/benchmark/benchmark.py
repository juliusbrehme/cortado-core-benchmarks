import copy
import time
import signal
from typing import List, Tuple, Optional
from cortado_core.utils.split_graph import SequenceGroup
from cortado_core.visual_query_language.query import create_query_instance, QueryType


# signal.alarm/SIGALRM only exist on Unix. On Windows we fall back to a
# cooperative wall-clock check between variants (see run_matching).
HAS_SIGALRM = hasattr(signal, "SIGALRM")


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Query execution timed out")


def run_matching(query, variants, query_type, deadline: Optional[float] = None):
    instance = create_query_instance(query, query_type)
    counter = 0
    for v in variants:
        counter += instance.match(v)
        # Cooperative timeout for platforms without SIGALRM (e.g. Windows): we
        # can't preempt, so we check the wall clock between variants instead.
        if deadline is not None and time.perf_counter() > deadline:
            raise TimeoutException("Query execution timed out")
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

        # Preemptive SIGALRM timeout on Unix; cooperative deadline elsewhere.
        use_signal = timeout_sec is not None and HAS_SIGALRM
        if use_signal:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_sec)

        start_time = time.perf_counter()
        deadline = start_time + timeout_sec if (timeout_sec is not None and not use_signal) else None
        try:
            for _ in range(iterations):
                run_matching(query_copy, variants, query_type, deadline=deadline)

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
            if use_signal:
                signal.alarm(0)

        timings.append(avg_time)

    return timings, match_count
