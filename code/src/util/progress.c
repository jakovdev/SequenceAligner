#include "util/progress.h"

#include "system/os.h"
#include "util/print.h"

static _Atomic(bool) p_running;
static _Atomic(u64) *p_progress;
static const char *p_message;
static u64 p_total;
static pthread_t p_thread;

static T_Func progress_monitor_worker(void *arg)
{
	(void)arg;
	print(M_PERCENT(0) "%s", p_message);

	while (atomic_load_relaxed(p_progress) < p_total) {
		usleep(100000);
		print(M_PROPORT(atomic_load_relaxed(p_progress) / p_total) "%s",
		      p_message);
	}

	print(M_PERCENT(100) "%s", p_message);
	T_Ret(NULL);
}

bool progress_start(_Atomic(u64) *progress, u64 total, const char *message)
{
	if (atomic_load(&p_running))
		goto p_thread_error;

	atomic_store(&p_running, true);
	p_progress = progress;
	p_message = message;
	p_total = total;

	pthread_create(&p_thread, NULL, progress_monitor_worker, NULL);
	if (p_thread)
		return true;

	atomic_store(&p_running, false);
p_thread_error:
	print_error_context("THREAD");
	print(M_NONE, ERR "Failed to create progress monitor thread");
	return false;
}

void progress_end(void)
{
	if (!atomic_load(&p_running)) {
		print(M_NONE, WARNING "No progress monitor is running");
		return;
	}

	pthread_join(p_thread, NULL);
	atomic_store(&p_running, false);
}
