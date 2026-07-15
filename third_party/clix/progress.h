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
 * progress_start(total_items, num_threads, "Processing items");
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
 * progress_start(rows * cols, num_threads, "Processing matrix");
 * double time_begin = current_time();
 * #pragma omp parallel
 * {
 *     #pragma omp for
 *     for (int i = 0; i < rows; i++) {
 *         for (int j = 0; j < cols; j++) {
 *             process_matrix_cell(i, j);
 *         }
 *         progress_add(cols);
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
 *
 * Source code:
 * @include progress.h
 */

#ifndef PROGRESS_H_
#define PROGRESS_H_

#include <stdbool.h>
#include <stddef.h>

/**
 * @brief Starts the progress monitor thread.
 *
 * Creates a background thread that updates the progress bar periodically.
 * Must be called from the main thread before starting parallel work.
 * Automatically adjusts update frequency based on @p total and @p threads.
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
void progress_end(void);

#ifdef PROGRESS_EXTERN_DISABLE
extern bool progress_disable;
#endif

#endif /* PROGRESS_H_ */
#if defined(PROGRESS_IMPLEMENTATION) && !defined(_PROGRESS_IMPLEMENTED)
#define _PROGRESS_IMPLEMENTED
bool progress_disable;

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
#define progress_bar(pct, msg)                 \
	do {                                   \
		printf("\r%s %d%%", msg, pct); \
		fflush(stdout);                \
	} while (0)
#endif /* progress_bar is a function */
#endif /* progress_bar */

#ifdef _MSC_VER
#define P_THREAD_LOCAL __declspec(thread)
#define P_ALIGNAS(a) __declspec(align(a))
#else
#define P_THREAD_LOCAL __thread
#define P_ALIGNAS(a) __attribute__((aligned(a)))
#endif

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#include <intrin.h>
typedef HANDLE p_thread_t;
typedef __int64 p_counter_t;
#define p_monitor_t unsigned __stdcall
#define P_ADD(ptr, val) _InterlockedExchangeAdd64((ptr), (val))
#define P_LOAD(ptr) _InterlockedExchangeAdd64((ptr), 0)
#define P_STORE(ptr, val) _InterlockedExchange64((ptr), (val))
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
#else
#include <pthread.h>
#include <time.h>
typedef pthread_t p_thread_t;
typedef long long p_counter_t;
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
#define P_WAIT_RESET()
#define P_WAIT_250()                                \
	struct timespec ts;                         \
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
#endif

static P_THREAD_LOCAL size_t t_done;

static P_ALIGNAS(64) p_counter_t p_done;
static p_thread_t p_monitor_thrd;

static size_t p_total;
static size_t p_update_limit;

static P_ALIGNAS(64) volatile p_counter_t p_running;

static p_monitor_t p_monitor(void *arg)
{
	const char *msg = arg;
	p_counter_t curr;
	progress_bar(0, msg);
	while (P_LOAD(&p_running)) {
		curr = P_LOAD(&p_done);
		progress_bar((int)(100 * curr / (p_counter_t)p_total), msg);
		P_WAIT_250();
	}
	progress_bar(100, msg);
	return P_SUCCESS;
}

bool progress_start(size_t total, int threads, const char *message)
{
	if (progress_disable)
		return true;
	if (P_LOAD(&p_running) || threads < 1)
		return false;
	if (!P_WAIT_INIT()) /* pthread_mutex/cond_init never error */
		return false;

	p_total = total ? total : 1;
	p_update_limit = total / ((size_t)threads * 100);
	if (!p_update_limit)
		p_update_limit = 1;

	P_WAIT_RESET();
	P_STORE(&p_done, 0);
	P_STORE(&p_running, 1);
	P_THREAD(p_monitor_thrd, p_monitor, (void *)message);
	if (p_monitor_thrd)
		return true;

	P_STORE(&p_running, 0);
	P_WAIT_DESTROY();
	return false;
}

void progress_flush(void)
{
	if (progress_disable || !t_done)
		return;

	P_ADD(&p_done, (p_counter_t)t_done);
	t_done = 0;
}

void progress_add(size_t amount)
{
	if (progress_disable || (t_done += amount) < p_update_limit)
		return;

	progress_flush();
}

void progress_end(void)
{
	if (progress_disable)
		return;
	if (!P_LOAD(&p_running))
		return;

	progress_flush();
	P_STORE(&p_done, (p_counter_t)p_total);
	P_STORE(&p_running, 0);
	P_WAIT_SIGNAL();
	P_JOIN(p_monitor_thrd);
	P_WAIT_DESTROY();
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
