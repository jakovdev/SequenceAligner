/*
CHANGES FROM ORIGINAL:
 - Silenced thrd_create warning
 - Simplified for mingw-only by removing everything else
 - Renamed file to threads.h for easier inclusion
===============================================================================
c11threads

Authors:
  John Tsiombikas <nuclear@member.fsf.org> - original POSIX threads wrapper
  Oliver Old <oliver.old@outlook.com> - win32 implementation

I place this piece of code in the public domain. Feel free to use as you see
fit. I'd appreciate it if you keep my name at the top of the code somewhere, but
whatever.

Main project site: https://github.com/jtsiomb/c11threads
*/
#ifndef C11THREADS_H_
#define C11THREADS_H_

#include <time.h>
#include <pthread.h>

typedef int (*thrd_start_t)(void *);
typedef void (*tss_dtor_t)(void *);

enum { mtx_plain, mtx_recursive, mtx_timed };
enum { thrd_success, thrd_timedout, thrd_busy, thrd_error, thrd_nomem };

#ifndef thread_local
#define thread_local _Thread_local
#endif

/* types */
typedef pthread_t thrd_t;
typedef pthread_mutex_t mtx_t;
typedef pthread_cond_t cnd_t;
typedef pthread_key_t tss_t;
typedef pthread_once_t once_flag;
#define ONCE_FLAG_INIT PTHREAD_ONCE_INIT

/* C11 threads over POSIX threads as thin wrapper functions */

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <sched.h>

struct thrd_args {
	thrd_start_t func;
	void *arg;
};

static inline void *_thrd_trampoline(void *ptr)
{
	struct thrd_args args = *(struct thrd_args *)ptr;
	free(ptr);
	return (void *)(intptr_t)args.func(args.arg);
}

/* ---- thread management ---- */
static inline int thrd_create(thrd_t *thr, thrd_start_t func, void *arg)
{
	struct thrd_args *args = malloc(sizeof(*args));
	if (!args)
		return thrd_nomem;
	*args = (struct thrd_args){ .func = func, .arg = arg };
	int res = pthread_create(thr, NULL, _thrd_trampoline, args);
	if (res != 0) {
		free(args);
		return res == ENOMEM ? thrd_nomem : thrd_error;
	}
	return thrd_success;
}

static inline void thrd_exit(int res)
{
	pthread_exit((void *)(intptr_t)res);
}

static inline int thrd_join(thrd_t thr, int *res)
{
	void *retval;
	if (pthread_join(thr, &retval) != 0)
		return thrd_error;
	if (res)
		*res = (int)(intptr_t)retval;
	return thrd_success;
}

static inline int thrd_detach(thrd_t thr)
{
	return pthread_detach(thr) == 0 ? thrd_success : thrd_error;
}

static inline thrd_t thrd_current(void)
{
	return pthread_self();
}

static inline int thrd_equal(thrd_t a, thrd_t b)
{
	return pthread_equal(a, b);
}

static inline int thrd_sleep(const struct timespec *req, struct timespec *rem)
{
	if (nanosleep(req, rem) < 0) {
		if (errno == EINTR)
			return -1;
		return -2;
	}
	return 0;
}

static inline void thrd_yield(void)
{
	sched_yield();
}

/* ---- mutexes ---- */

static inline int mtx_init(mtx_t *mtx, int type)
{
	int res;
	pthread_mutexattr_t attr;
	pthread_mutexattr_init(&attr);
	if (type & mtx_timed) {
#ifdef PTHREAD_MUTEX_TIMED_NP
		pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_TIMED_NP);
#else
		pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_NORMAL);
#endif
	}
	if (type & mtx_recursive)
		pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
	res = pthread_mutex_init(mtx, &attr) == 0 ? thrd_success : thrd_error;
	pthread_mutexattr_destroy(&attr);
	return res;
}

static inline void mtx_destroy(mtx_t *mtx)
{
	pthread_mutex_destroy(mtx);
}

static inline int mtx_lock(mtx_t *mtx)
{
	int res = pthread_mutex_lock(mtx);
	return res == 0 ? thrd_success : thrd_error;
}

static inline int mtx_trylock(mtx_t *mtx)
{
	int res = pthread_mutex_trylock(mtx);
	if (res == EBUSY)
		return thrd_busy;
	return res == 0 ? thrd_success : thrd_error;
}

static inline int mtx_timedlock(mtx_t *mtx, const struct timespec *ts)
{
	int res = 0;
	if ((res = pthread_mutex_timedlock(mtx, ts)) == ETIMEDOUT)
		return thrd_timedout;
	return res == 0 ? thrd_success : thrd_error;
}

static inline int mtx_unlock(mtx_t *mtx)
{
	return pthread_mutex_unlock(mtx) == 0 ? thrd_success : thrd_error;
}

/* ---- condition variables ---- */

static inline int cnd_init(cnd_t *cond)
{
	return pthread_cond_init(cond, 0) == 0 ? thrd_success : thrd_error;
}

static inline void cnd_destroy(cnd_t *cond)
{
	pthread_cond_destroy(cond);
}

static inline int cnd_signal(cnd_t *cond)
{
	return pthread_cond_signal(cond) == 0 ? thrd_success : thrd_error;
}

static inline int cnd_broadcast(cnd_t *cond)
{
	return pthread_cond_broadcast(cond) == 0 ? thrd_success : thrd_error;
}

static inline int cnd_wait(cnd_t *cond, mtx_t *mtx)
{
	return pthread_cond_wait(cond, mtx) == 0 ? thrd_success : thrd_error;
}

static inline int cnd_timedwait(cnd_t *cond, mtx_t *mtx,
				const struct timespec *ts)
{
	int res;
	if ((res = pthread_cond_timedwait(cond, mtx, ts)) != 0)
		return res == ETIMEDOUT ? thrd_timedout : thrd_error;
	return thrd_success;
}

/* ---- thread-specific data ---- */

static inline int tss_create(tss_t *key, tss_dtor_t dtor)
{
	return pthread_key_create(key, dtor) == 0 ? thrd_success : thrd_error;
}

static inline void tss_delete(tss_t key)
{
	pthread_key_delete(key);
}

static inline int tss_set(tss_t key, void *val)
{
	return pthread_setspecific(key, val) == 0 ? thrd_success : thrd_error;
}

static inline void *tss_get(tss_t key)
{
	return pthread_getspecific(key);
}

/* ---- misc ---- */

static inline void call_once(once_flag *flag, void (*func)(void))
{
	pthread_once(flag, func);
}

#endif /* C11THREADS_H_ */
