#include "util/progress.h"

#include <stdalign.h>
#include <stdatomic.h>
#include <threads.h>

#include "util/args.h"
#include "util/print.h"

static thread_local size_t t_done;

static alignas(64) atomic_size_t p_done;
static thrd_t p_monitor_thrd;

static size_t p_total;
static size_t p_update_limit;
static const char *p_message;

static atomic_bool p_running;
static bool p_disable;

static mtx_t p_mutex;
static cnd_t p_cond;

static int p_monitor(void *arg)
{
	(void)arg;
	ppercent(0, "%s", p_message);

	struct timespec timeout;

	size_t current = 0;
	while (atomic_load_explicit(&p_running, memory_order_acquire)) {
		current = atomic_load_explicit(&p_done, memory_order_relaxed);
		pproportc(current / p_total, "%s", p_message);

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

	ppercent(100, "%s", p_message);
	return thrd_success;
}

bool progress_start(size_t total, int threads, const char *message)
{
	if (p_disable)
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
	perr("Failed to create progress bar monitor thread");
	return false;
}

void progress_flush(void)
{
	if (p_disable || !t_done)
		return;

	atomic_fetch_add_explicit(&p_done, t_done, memory_order_relaxed);
	t_done = 0;
}

void progress_add(size_t amount)
{
	if (p_disable || (t_done += amount) < p_update_limit)
		return;

	progress_flush();
}

void progress_end(void)
{
	if (p_disable)
		return;

	if (!atomic_load_explicit(&p_running, memory_order_relaxed)) {
		pdev("Tried to end non-running progress monitor");
		perr("Internal error during progress bar monitor thread cleanup");
		return;
	}

	progress_flush();
	atomic_store_explicit(&p_done, p_total, memory_order_relaxed);
	atomic_store_explicit(&p_running, false, memory_order_release);
	cnd_signal(&p_cond);
	thrd_join(p_monitor_thrd, NULL);
	cnd_destroy(&p_cond);
	mtx_destroy(&p_mutex);
}

ARG_EXTERN(disable_write);

ARGUMENT(disable_progress) = {
	.opt = 'P',
	.lopt = "no-progress",
	.help = "Disable progress bars",
	.set = &p_disable,
	.help_order = ARG_ORDER_AFTER(disable_write),
};
