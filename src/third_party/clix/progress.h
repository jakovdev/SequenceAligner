/* SPDX-License-Identifier: MIT */
/**
 * @page progress.h
 * @author Jakov Dragičević (github:jakovdev)
 * @copyright MIT License.
 * @brief Progress bar display.
 * @details
 * Supports both Windows and common POSIX systems and ANSI C/C++ or later.
 * @note If you get declaration errors on POSIX systems define @c _POSIX_C_SOURCE
 * to >= @c 199309L before including anything in the implementer source file
 *
 * @attention Limitations:
 * - Only one progress monitor can be active at a time.
 * - Only prints whole/integer percentage points (0%, 1%, ... 99%, 100%).
 * - Calls @ref progress_bar in 250ms intervals except after @ref progress_end.
 *
 * @see @subpage progress_intro "Progress bar display"
 *
 * Source code:
 * @include progress.h
 */

/**
 * @page progress_intro Progress bar display
 *
 * # Overview
 * @ref progress.h "progress.h" provides a header-only progress bar display
 * module for C codebases.
 *
 * # Feature Highlights
 * - Low/no overhead
 * - Customizeable "display" functionality
 *
 * # Quick Start
 * 1. Before parallel region in main thread call @ref progress_start.
 * 2. Inside parallel region in the outer-most loop call @ref progress_add.
 * 3. After parallel region in main thread call @ref progress_end.
 *
 * You can also use this for non-CPU/parallel workloads, in which case:
 * 1. Call @ref progress_start with @p threads = 1
 * 2. Call @ref progress_add with actual amount of work done out of @p total
 * 3. Call @ref progress_end when done.
 *
 * # Example #1
 * @code{.c}
 * progress_start(total_items, num_threads, "Processing items");
 * #pragma omp parallel
 * {
 *     #pragma omp for
 *     for (int i = 0; i < total_items; i++) {
 *         process_item(i);
 *         progress_add(1);
 *     }
 * }
 * progress_end();
 * @endcode
 *
 * # Example #2
 * @code{.c}
 * progress_start(rows * cols, num_threads, "Processing matrix");
 * double time_begin = current_time();
 * #pragma omp parallel
 * {
 *     #pragma omp for
 *     for (int i = 0; i < rows; i++) {
 *         for (int j = 0; j < cols; j++) {
 *             process_matrix_cell(i, j);
 *         }
 *         progress_add(cols); // minimize progress_add() calls
 *     }
 * }
 * double time_end = current_time(); // Do not include progress_end() time
 * progress_end(); // prints "\rProcessing matrix 100%" by default
 * printf("Matrix processing time: %.2f", time_end - time_begin);
 * @endcode
 *
 * # Example #3
 * @code{.c}
 * progress_start(total, 1, "Kernel progress");
 * while (start < total)
 * {
 *     size_t batch = start + BATCH_SIZE > total ? total - start : BATCH_SIZE;
 *     gpu_kernel_batch(start, batch, kernel, args);
 *     gpu_kernel_wait();
 *     progress_add(batch);
 *     start += batch;
 * }
 * progress_end();
 * @endcode
 *
 * # Docs Navigation
 * - @ref progress_core "Core API"
 * - @ref progress_customizable "Customization Points"
 */

#ifndef PROGRESS_H_
#define PROGRESS_H_

#include <stddef.h>

/** @defgroup progress_core progress.h: Core API */
/** @defgroup progress_customizable progress.h: Customization Points */

/** @addtogroup progress_core
 * @brief Progress monitor creation and manipulation.
 */

/** @ingroup progress_core
 * @brief Starts the progress monitor thread.
 *
 * Creates a background thread that updates the progress bar every 250ms.
 * Must be called from the main thread before starting parallel work.
 * Automatically adjusts counter update frequency based on @p total, @p threads.
 *
 * @param total Total number of work units to complete.
 * @param threads Number of threads that will be adding to counter.
 * @param message Description displayed alongside the progress bar.
 * @return 0/false if thread creation fails or already running, else 1/true.
 */
int progress_start(size_t total, int threads, const char *message);

/** @ingroup progress_core
 * @brief Increments the internal thread-local progress counter.
 *
 * Must be called from all threads that perform work out of @p total.
 * Progress is thread-local to minimize atomic operations on global counter.
 * Does not guarantee immediate update if @p amount too small.
 * Recommended to call in the outer-most loop with highest @p amount possible.
 *
 * @param amount Number of thread-local work units completed out of @p total.
 */
void progress_add(size_t amount);

/** @ingroup progress_core
 * @brief Displays 100% and stops the progress monitor thread.
 *
 * Must be called from the main thread after all parallel work completes before
 * starting another progress monitor.
 */
void progress_end(void);

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L
typedef bool progress_bool_t;
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#include <stdbool.h>
typedef bool progress_bool_t;
#elif defined(__cplusplus)
typedef bool progress_bool_t;
#else
typedef int progress_bool_t;
#endif

/** @addtogroup progress_customizable
 * @brief Customization points for overriding default behaviors.
 * @details
 * Define the following macros before including @ref progress.h to override
 * default implementations.
 */

#ifdef PROGRESS_EXTERN_DISABLE
/** @ingroup progress_customizable
 * @brief Set to true/1 to ignore function calls.
 * @note Available when @c PROGRESS_EXTERN_DISABLE is defined before including
 * @ref progress.h.
 */
extern progress_bool_t progress_disable;
#endif

#endif /* PROGRESS_H_ */
#if defined(PROGRESS_IMPLEMENTATION) && !defined(_PROGRESS_IMPLEMENTED)
#define _PROGRESS_IMPLEMENTED
progress_bool_t progress_disable;

#ifndef PROGRESS_PRINT_H
#define PROGRESS_PRINT_H 0
#include <stdio.h>
#else
#ifndef PRINT_H_
#error "Include print.h before including progress.h"
#endif /* PRINT_H_ */
#undef PROGRESS_PRINT_H
#define PROGRESS_PRINT_H 1
#endif /* PROGRESS_PRINT_H */

#ifndef progress_bar
#if !PROGRESS_PRINT_H
/** @ingroup progress_customizable
 * @brief The "printing" function called by progress monitor.
 * @remark Overridable before including @ref progress.h.
 */
#define progress_bar(pct, msg)         \
	printf("\r%s %d%%", msg, pct); \
	fflush(stdout);
#endif /* progress_bar is a function in print.h */
#endif /* progress_bar */

#ifdef _MSC_VER
#define P_THREAD_LOCAL __declspec(thread)
#define P_ALIGNAS(a) __declspec(align(a))
#else
#define P_THREAD_LOCAL __thread
#define P_ALIGNAS(a) __attribute__((aligned(a)))
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <process.h>
typedef HANDLE p_thread_t;
#define p_monitor_t unsigned __stdcall
#define _P_CAST(x) (volatile __int64 *)(x)
#define P_ADD(ptr, val) _InterlockedExchangeAdd64(_P_CAST(ptr), (val))
#define P_LOAD(ptr) _InterlockedExchangeAdd64(_P_CAST(ptr), 0)
#define P_STORE(ptr, val) _InterlockedExchange64(_P_CAST(ptr), (val))
#define P_THREAD(t, f, a) t = (HANDLE)_beginthreadex(NULL, 0, f, a, 0, NULL)
#define P_JOIN(t)                         \
	WaitForSingleObject(t, INFINITE); \
	CloseHandle(t);
#define P_SUCCESS 0
static HANDLE p_wait_event;
#define P_WAIT_INIT() (p_wait_event = CreateEvent(NULL, TRUE, FALSE, NULL))
#define P_WAIT_DESTROY() CloseHandle(p_wait_event)
#define P_WAIT_250() WaitForSingleObject(p_wait_event, 250)
#define P_WAIT_SIGNAL() SetEvent(p_wait_event)
#define P_WAIT_RESET() ResetEvent(p_wait_event)
#define P_WAIT_TIME_INIT()
#else
#include <pthread.h>
#include <time.h>
typedef pthread_t p_thread_t;
typedef void *p_monitor_t;
#define P_ADD(ptr, val) __atomic_fetch_add((ptr), (val), __ATOMIC_RELAXED)
#define P_LOAD(ptr) __atomic_load_n((ptr), __ATOMIC_ACQUIRE)
#define P_STORE(ptr, val) __atomic_store_n((ptr), (val), __ATOMIC_RELEASE)
#define P_THREAD(t, f, a) pthread_create(&t, NULL, f, a)
#define P_JOIN(t) pthread_join(t, NULL)
#define P_SUCCESS NULL
static pthread_mutex_t p_mtx;
static pthread_cond_t p_cv;
#define P_WAIT_INIT() \
	(!pthread_mutex_init(&p_mtx, NULL) && !pthread_cond_init(&p_cv, NULL))
#define P_WAIT_DESTROY()               \
	pthread_mutex_destroy(&p_mtx); \
	pthread_cond_destroy(&p_cv)
#define P_WAIT_250()                                \
	clock_gettime(CLOCK_REALTIME, &ts);         \
	ts.tv_nsec += 250000000L;                   \
	if (ts.tv_nsec >= 1000000000L) {            \
		ts.tv_sec++;                        \
		ts.tv_nsec -= 1000000000L;          \
	}                                           \
	pthread_mutex_lock(&p_mtx);                 \
	pthread_cond_timedwait(&p_cv, &p_mtx, &ts); \
	pthread_mutex_unlock(&p_mtx)
#define P_WAIT_SIGNAL()             \
	pthread_mutex_lock(&p_mtx); \
	pthread_cond_signal(&p_cv); \
	pthread_mutex_unlock(&p_mtx)
#define P_WAIT_RESET()
#define P_WAIT_TIME_INIT() struct timespec ts;
#endif

static P_THREAD_LOCAL size_t t_done;

static P_ALIGNAS(64) size_t p_done;
static p_thread_t p_monitor_thrd;

static size_t p_total;
static size_t t_limit;

static P_ALIGNAS(64) volatile size_t p_running;

static p_monitor_t p_monitor(void *arg)
{
	P_WAIT_TIME_INIT()
	const char *msg = (const char *)arg;
	progress_bar(0, msg);
	while (P_LOAD(&p_running)) {
		progress_bar((int)(100 * P_LOAD(&p_done) / p_total), msg);
		P_WAIT_250();
	}
	progress_bar(100, msg);
	return P_SUCCESS;
}

int progress_start(size_t total, int threads, const char *message)
{
	if (progress_disable)
		return 1;
	if (P_LOAD(&p_running) || threads < 1 || !P_WAIT_INIT())
		return 0;

	p_total = total ? total : 1;
	t_limit = total / ((size_t)threads * 100);
	if (!t_limit)
		t_limit = 1;

	P_WAIT_RESET();
	P_STORE(&p_done, 0);
	P_STORE(&p_running, 1);
	P_THREAD(p_monitor_thrd, p_monitor, (void *)message);
	if (p_monitor_thrd)
		return 1;

	P_STORE(&p_running, 0);
	P_WAIT_DESTROY();
	return 0;
}

static void progress_flush(void)
{
	if (t_done) {
		P_ADD(&p_done, t_done);
		t_done = 0;
	}
}

void progress_add(size_t a)
{
	if (!progress_disable && P_LOAD(&p_running) && (t_done += a) >= t_limit)
		progress_flush();
}

void progress_end(void)
{
	if (!progress_disable && P_LOAD(&p_running)) {
		progress_flush();
		P_STORE(&p_done, p_total);
		P_STORE(&p_running, 0);
		P_WAIT_SIGNAL();
		P_JOIN(p_monitor_thrd);
		P_WAIT_DESTROY();
	}
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
