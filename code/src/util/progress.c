#include "util/progress.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

typedef HANDLE pthread_t;

#define T_Func DWORD WINAPI
#define T_Ret(x) return (DWORD)(size_t)(x)
#define pthread_join(thread_id, _) WaitForSingleObject(thread_id, INFINITE)
#define usleep(microseconds) Sleep((microseconds) / 1000)

static int pthread_create(pthread_t *restrict thread, void *restrict _,
			  LPTHREAD_START_ROUTINE fn, void *restrict arg)
{
	(void)_;
	*thread = CreateThread(NULL, 0, fn, arg, 0, NULL);
	return *thread ? 0 : -1;
}
#else
#include <pthread.h>
#include <unistd.h>

typedef void *T_Func;

#define T_Ret(x) return (x)
#endif

#define atomic_load_relaxed(p) atomic_load_explicit((p), memory_order_relaxed)
#define atomic_add_relaxed(p, v) \
	atomic_fetch_add_explicit((p), (v), memory_order_relaxed)

#include "util/print.h"

static _Atomic(bool) p_running;
static _Atomic(s64) *p_progress;
static const char *p_message;
static s64 p_total;
static pthread_t p_thread;

static T_Func p_monitor(void *arg)
{
	(void)arg;
	ppercent(0, "%s", p_message);

	while (atomic_load_relaxed(p_progress) < p_total) {
		usleep(100000);
		pproportc(atomic_load_relaxed(p_progress) / p_total, "%s",
			  p_message);
	}

	ppercent(100, "%s", p_message);
	T_Ret(NULL);
}

bool progress_start(_Atomic(s64) *progress, s64 total, const char *message)
{
	if (atomic_load(&p_running))
		goto p_thread_error;

	atomic_store(&p_running, true);
	p_progress = progress;
	p_message = message;
	p_total = total;

	if (pthread_create(&p_thread, NULL, p_monitor, NULL) == 0)
		return true;

	atomic_store(&p_running, false);
p_thread_error:
	perr("Failed to create progress bar monitor thread");
	return false;
}

void progress_end(void)
{
	if (!atomic_load(&p_running)) {
		pdev("Tried to end non-running progress monitor");
		perr("Internal error during progress bar monitor thread cleanup");
		return;
	}

	pthread_join(p_thread, NULL);
	atomic_store(&p_running, false);
}
