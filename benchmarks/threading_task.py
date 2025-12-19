import threading
import time


def partial_sum(start, end):
    """Calculates a partial sum of a simple series."""
    total = 0
    for i in range(start, end):
        total += (i * i) ** 0.5
    return total


def run_threaded_sum(num_threads, work_size):
    """
    Runs a CPU-bound calculation across multiple threads and measures the
    wall-clock time.
    """
    threads = []
    chunk_size = work_size // num_threads
    results = [0] * num_threads

    def worker(index, start, end):
        results[index] = partial_sum(start, end)

    start_time = time.perf_counter()

    for i in range(num_threads):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_threads - 1 else work_size
        thread = threading.Thread(target=worker, args=(i, start, end))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    end_time = time.perf_counter()
    total_sum = sum(results)

    return {
        "total_sum": total_sum,
        "duration_s": end_time - start_time,
        "threads": num_threads,
    }
