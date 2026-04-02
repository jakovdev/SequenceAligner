// SPDX-License-Identifier: MIT
/**
 * @page progress.h
 * @author Jakov Dragičević (github:jakovdev)
 * @copyright MIT License.
 * @attention This library is work-in-progress. Do not use if not meaning to contribute.
 * @brief Progress bar display for parallel workloads.
 *
 * Before the parallel region (main thread):
 * 1. Call @ref progress_start.
 *
 * Inside the parallel region (all worker threads):
 * 2. Call @ref progress_add inside the outer-most loop.
 * 3. Call @ref progress_flush outside the outer-most loop.
 *
 * After the parallel region (main thread):
 * 4. Call @ref progress_end.
 *
 * Example #1:
 * @code{.c}
 * progress_start(total_items, "Processing items");
 * #pragma omp parallel
 * {
 *     #pragma omp for
 *     for (int i = 0; i < total_items; i++) {
 *         process_item(i);
 *         progress_add(1);
 *     }
 *     progress_flush();
 * }
 * progress_end();
 * @endcode
 *
 * Example #2:
 * @code{.c}
 * progress_start(total_items, "Processing matrix");
 * double time_begin = current_time();
 * #pragma omp parallel
 * {
 *     #pragma omp for
 *     for (int i = 0; i < rows; i++) {
 *         for (int j = 0; j < cols; j++) {
 *             process_matrix_cell(i, j);
 *         }
 *         // items_per_row depends on the inner loop
 *         const s64 items_per_row = cols;
 *         progress_add(items_per_row);
 *     }
 *     progress_flush();
 * }
 * double time_end = current_time();
 * progress_end();
 * printf("Matrix processing time: %.2f", time_end - time_begin);
 * @endcode
 *
 * Limitations:
 * - Only one progress monitor thread can be active at a time.
 * - @ref progress_end may cause slight delays from mutex contention.
 * - Requires C11 (atomics) to build the source file. On Windows https://github.com/tinycthread/tinycthread is recommended.
 *
 * Source code:
 * @include progress.h
 */

#ifndef PROGRESS_H
#define PROGRESS_H

#include <stdbool.h>
#include <stddef.h>

/**
 * @brief Starts the progress monitor thread.
 *
 * Creates a background thread that updates the progress bar periodically.
 * Must be called from the main thread before starting parallel work.
 * Automatically adjusts update frequency based on @p total.
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
 * Must be called from all threads that perform work out of @p total.
 * Progress is batched locally to minimize atomic operations.
 * Does not guarantee immediate update if @p amount too small.
 * Recommended to call in the outer-most loop with highest @p amount possible.
 *
 * @param amount Number of thread-local work units completed out of @p total.
 */
void progress_add(size_t amount);

/**
 * @brief Flushes thread-local progress.
 *
 * Call once per thread after it finishes its work so that progress bar
 * reaches 100% as some @p amount may still be buffered.
 */
void progress_flush(void);

/**
 * @brief Stops the progress monitor thread.
 *
 * Must be called from the main thread after all parallel work completes.
 * May be slightly delayed due to mutex contention.
 */
bool progress_end(void);

#ifdef PROGRESS_EXTERN_DISABLE
extern bool progress_disable;
#endif

#endif /* PROGRESS_H */
#if defined(PROGRESS_IMPLEMENTATION) && !defined(_PROGRESS_IMPLEMENTED)
#define _PROGRESS_IMPLEMENTED

#include <stdalign.h>
#include <stdatomic.h>
#include <threads.h>

#ifndef PROGRESS_PRINT_H
#define PROGRESS_PRINT_H 0
#include <stdio.h>
#else
#ifndef PRINT_H
#error "Include print.h before including progress.h"
#endif /* PRINT_H */
#undef PROGRESS_PRINT_H
#define PROGRESS_PRINT_H 1
#endif /* PROGRESS_PRINT_H */

#ifndef progress_bar
#if !PROGRESS_PRINT_H
#define progress_bar(pct, ...)           \
	do {                             \
		printf("\r%3d%% ", pct); \
		printf(__VA_ARGS__);     \
		fflush(stdout);          \
	} while (0)
#endif /* progress_bar is a function */
#endif /* progress_bar */

#ifndef progress_err
#if PROGRESS_PRINT_H
#define progress_err(...) perr(__VA_ARGS__)
#else /* Default */
#define progress_err(...) fprintf(stderr, __VA_ARGS__)
#endif /* PROGRESS_PRINT_H */
#endif /* progress_err */

#ifndef progress_dev
#if PROGRESS_PRINT_H
#define progress_dev(...) pdev(__VA_ARGS__)
#else /* Default */
#define progress_dev(...) fprintf(stderr, __VA_ARGS__)
#endif /* PROGRESS_PRINT_H */
#endif /* progress_dev */

static _Thread_local size_t t_done;

static _Alignas(64) _Atomic(size_t) p_done;
static thrd_t p_monitor_thrd;

static size_t p_total;
static size_t p_update_limit;
static const char *p_message;

static _Atomic(bool) p_running;
static bool progress_disable;

static mtx_t p_mutex;
static cnd_t p_cond;

static int p_monitor(void *arg)
{
	(void)arg;
	progress_bar(0, "%s", p_message);

	struct timespec timeout;

	size_t current = 0;
	while (atomic_load_explicit(&p_running, memory_order_acquire)) {
		current = atomic_load_explicit(&p_done, memory_order_relaxed);
		progress_bar((int)(100 * current / p_total), "%s", p_message);

		if (timespec_get(&timeout, TIME_UTC) == 0) {
			timeout.tv_sec = 0;
			timeout.tv_nsec = 0;
		}
		timeout.tv_nsec += 250000000;
		if (timeout.tv_nsec >= 1000000000) {
			timeout.tv_sec++;
			timeout.tv_nsec -= 1000000000;
		}

		mtx_lock(&p_mutex);
		cnd_timedwait(&p_cond, &p_mutex, &timeout);
		mtx_unlock(&p_mutex);
	}

	progress_bar(100, "%s", p_message);
	return thrd_success;
}

bool progress_start(size_t total, int threads, const char *message)
{
	if (progress_disable)
		return true;

	if (atomic_load_explicit(&p_running, memory_order_relaxed))
		goto p_monitor_running_error;

	atomic_store_explicit(&p_running, true, memory_order_relaxed);
	atomic_store_explicit(&p_done, 0, memory_order_relaxed);
	p_message = message;
	p_total = total;
	if (!p_total)
		p_total = 1;
	p_update_limit = total / ((size_t)threads * 100);
	if (!p_update_limit)
		p_update_limit = 1;

	if (mtx_init(&p_mutex, mtx_plain) != thrd_success)
		goto p_monitor_mtx_error;
	if (cnd_init(&p_cond) != thrd_success)
		goto p_monitor_cnd_error;

	if (thrd_create(&p_monitor_thrd, p_monitor, NULL) == thrd_success)
		return true;

	cnd_destroy(&p_cond);
p_monitor_cnd_error:
	mtx_destroy(&p_mutex);
p_monitor_mtx_error:
	atomic_store_explicit(&p_running, false, memory_order_relaxed);
p_monitor_running_error:
	progress_err("Failed to create a progress bar display");
	return false;
}

void progress_flush(void)
{
	if (progress_disable || !t_done)
		return;

	atomic_fetch_add_explicit(&p_done, t_done, memory_order_relaxed);
	t_done = 0;
}

void progress_add(size_t amount)
{
	if (progress_disable || (t_done += amount) < p_update_limit)
		return;

	progress_flush();
}

bool progress_end(void)
{
	if (progress_disable)
		return true;

	if (!atomic_load_explicit(&p_running, memory_order_relaxed)) {
		progress_dev("Tried to end non-running progress monitor");
		progress_err("Internal error during progress bar display");
		return false;
	}

	progress_flush();
	atomic_store_explicit(&p_done, p_total, memory_order_relaxed);
	atomic_store_explicit(&p_running, false, memory_order_release);
	cnd_signal(&p_cond);
	thrd_join(p_monitor_thrd, NULL);
	cnd_destroy(&p_cond);
	mtx_destroy(&p_mutex);
	return true;
}

#endif /* PROGRESS_IMPLEMENTATION */

/*
progress.h
https://github.com/jakovdev/clix/
Copyright (c) 2026 Jakov Dragičević
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
