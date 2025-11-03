#include "util/progress.h"

#include "system/arch.h"
#include "util/print.h"

static _Atomic(bool) p_running;
static _Atomic(size_t) *p_progress;
static const char *p_message;
static size_t p_total;
static pthread_t p_thread;

static T_Func progress_monitor_worker(void *arg)
{
	(void)arg;
	int percentage = 0;
	print(M_PERCENT(percentage) "%s", p_message);

	while (atomic_load_relaxed(p_progress) < p_total) {
		usleep(100000);
		percentage =
			(int)(100 * atomic_load_relaxed(p_progress) / p_total);
		print(M_PERCENT(percentage) "%s", p_message);
	}

	if (percentage < 100)
		print(M_PERCENT(100) "%s", p_message);

	T_Ret(NULL);
}

bool progress_start(_Atomic(size_t) *progress, size_t total,
		    const char *message)
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
	print(M_NONE, ERROR "Failed to create progress monitor thread");
	return false;
}

void progress_end(void)
{
	if (!atomic_load(&p_running)) {
		print_error_context("THREAD");
		print(M_NONE, WARNING "No progress monitor is running");
		return;
	}

	pthread_join(p_thread, NULL);
	atomic_store(&p_running, false);
}
