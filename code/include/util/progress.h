#pragma once
#ifndef UTIL_PROGRESS_H
#define UTIL_PROGRESS_H

/**
  * @file progress.h
  * @brief Progress bar display for parallel workloads.
  * 
  * Basic usage:
  * 
  * Before the parallel region (main thread):
  * 
  * 1. Call `progress_start`.
  * 
  * Inside the parallel region (all worker threads):
  * 
  * 2. Call `progress_add` inside the outer-most loop.
  * 
  * 3. Call `progress_flush` outside the outer-most loop.
  * 
  * After the parallel region (main thread):
  * 
  * 4. Call `progress_end`.
  * 
  * Example #1:
  * @code
  * 	progress_start(total_items, "Processing items");
  * 	#pragma omp parallel
  * 	{
  * 		#pragma omp for
  * 		for (int i = 0; i < total_items; i++) {
  * 			process_item(i);
  * 			progress_add(1);
  * 		}
  * 		progress_flush();
  * 	}
  * 	progress_end(); @endcode
  * 
  * Example #2:
  * @code
  * 	progress_start(total_items, "Processing matrix");
  * 	double time_begin = current_time();
  * 	#pragma omp parallel
  * 	{
  * 		#pragma omp for
  * 		for (int i = 0; i < rows; i++) {
  * 			for (int j = 0; j < cols; j++) {
  * 				process_matrix_cell(i, j);
  * 			}
  * 			// items_per_row depends on the inner loop
  * 			const s64 items_per_row = cols;
  * 			progress_add(items_per_row);
  * 		}
  * 		progress_flush();
  * 	}
  * 	double time_end = current_time();
  * 	progress_end();
  * 	printf("Matrix processing time: %.2f", time_end - time_begin); @endcode
  * 
  * Limitations:
  * 
  * - Only one progress monitor thread can be active at a time.
  * 
  * - `progress_end` may cause slight delays from mutex contention.
  * 
  * - Requires C11 (atomics) to build the source file.
  */

#include <stdbool.h>
#include <stddef.h>

/**
  * @brief Starts the progress monitor thread.
  * 
  * Creates a background thread that updates the progress bar periodically.
  * Must be called from the main thread before starting parallel work.
  * Automatically adjusts update frequency based on `total`
  * 
  * @param total Total number of work units to complete.
  * @param threads Number of threads that will be performing work.
  * @param message Description displayed alongside the progress bar.
  * @return false if thread creation fails or already running, true otherwise.
  */
bool progress_start(size_t total, int threads, const char *message);

/**
  * @brief Increments the progress counter.
  * 
  * Must be called from all threads that perform work out of `total`.
  * Progress is batched locally to minimize atomic operations.
  * Does not guarantee immediate update if `amount` too small.
  * Recommended to call in the outer-most loop with highest `amount` possible.
  * 
  * @param amount Number of thread-local work units completed out of `total`.
  */
void progress_add(size_t amount);

/**
  * @brief Flushes thread-local progress.
  * 
  * Call once per thread after it finishes its work so that progress bar
  * reaches 100% as some `amount` may still be buffered.
  */
void progress_flush(void);

/**
  * @brief Stops the progress monitor thread.
  * 
  * Must be called from the main thread after all parallel work completes.
  * May be slightly delayed due to mutex contention.
  */
void progress_end(void);

#endif /* UTIL_PROGRESS_H */
