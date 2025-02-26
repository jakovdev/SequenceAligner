from typing import List, Callable, Any, Tuple, Dict, Iterator, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from functools import partial
import multiprocessing
import logging
import numpy as np

logger = logging.getLogger(__name__)


def get_optimal_workers(requested: int = 0) -> int:
    if requested <= 0:
        # Use CPU count - 1 to leave one core for system processes
        return max(1, multiprocessing.cpu_count() - 1)
    else:
        return min(requested, multiprocessing.cpu_count())


def calculate_optimal_chunk_size(total_items: int, n_workers: int = 0) -> int:
    n_workers = get_optimal_workers(n_workers)
    target_n_chunks = n_workers * 4
    chunk_size = max(1, total_items // target_n_chunks)
    chunk_size = min(max(chunk_size, 64), 8192)  # Capped for memory usage
    return min(chunk_size, total_items)


def parallel_process(
    func: Callable,
    items: List[Any],
    n_workers: int = 0,
    use_tqdm: bool = True,
    desc: str = "Processing",
    **kwargs,
) -> List[Any]:
    n_workers = get_optimal_workers(n_workers)
    results = []

    if kwargs:
        func = partial(func, **kwargs)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_item = {
            executor.submit(func, item): i for i, item in enumerate(items)
        }

        futures = as_completed(future_to_item)
        if use_tqdm:
            futures = tqdm(futures, total=len(items), desc=desc)

        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                idx = future_to_item[future]
                logger.error(f"Error processing item {idx}: {exc}")
                results.append(None)

    return results


def process_matrix_chunks(
    func: Callable,
    data: List[Any],
    # chunk_size: int,
    n_workers: int = 0,
    symmetric: bool = True,
    use_tqdm: bool = True,
    desc: str = "Processing chunks",
    **kwargs,
) -> np.ndarray:
    n_items = len(data)
    n_workers = get_optimal_workers(n_workers)
    chunk_size = calculate_optimal_chunk_size(n_items, n_workers)
    result_matrix = np.zeros((n_items, n_items), dtype=np.float32)

    if symmetric:
        np.fill_diagonal(result_matrix, 1.0)

    chunks = []
    for i in range(0, n_items, chunk_size):
        i_end = min(i + chunk_size, n_items)

        if symmetric:
            j_start = i
        else:
            j_start = 0

        for j in range(j_start, n_items, chunk_size):
            j_end = min(j + chunk_size, n_items)
            chunks.append((i, i_end, j, j_end))

    process_func = partial(_process_matrix_chunk, func=func, data=data, **kwargs)

    with tqdm(total=len(chunks), desc=desc, disable=not use_tqdm) as pbar:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_func, chunk): chunk for chunk in chunks}
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    i_start, i_end, j_start, j_end, chunk_result = future.result()
                    result_matrix[i_start:i_end, j_start:j_end] = chunk_result
                    if symmetric and i_start != j_start:
                        result_matrix[j_start:j_end, i_start:i_end] = chunk_result.T

                    pbar.update(1)
                except Exception as e:
                    i_start, i_end, j_start, j_end = chunk
                    logger.error(f"Error processing chunk {chunk}: {e}")

    return result_matrix


def _process_matrix_chunk(
    chunk: Tuple[int, int, int, int], func: Callable, data: List[Any], **kwargs
) -> Tuple[int, int, int, int, np.ndarray]:
    i_start, i_end, j_start, j_end = chunk
    data_i = data[i_start:i_end]
    data_j = data[j_start:j_end]
    chunk_result = func(data_i, data_j, i_start, i_end, j_start, j_end, **kwargs)
    return i_start, i_end, j_start, j_end, chunk_result
