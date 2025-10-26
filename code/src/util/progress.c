#include "util/progress.h"

#include "system/arch.h"
#include "util/print.h"

static _Atomic(bool) p_running = false;
static _Atomic(size_t)* p_progress = NULL;
static const char* p_message = NULL;
static size_t p_total = 0;
static pthread_t p_thread = 0;

static T_Func
progress_monitor_worker(void* arg)
{
    (void)arg;
    int percentage = 0;
    print(PROGRESS, MSG_PERCENT(percentage), "%s", p_message);

    while (atomic_load_explicit(p_progress, memory_order_relaxed) < p_total)
    {
        usleep(100000);
        percentage = (int)(100 * atomic_load_explicit(p_progress, memory_order_relaxed) / p_total);
        print(PROGRESS, MSG_PERCENT(percentage), "%s", p_message);
    }

    if (percentage < 100)
    {
        print(PROGRESS, MSG_PERCENT(100), "%s", p_message);
    }

    T_Ret(NULL);
}

bool
progress_start(_Atomic(size_t)* progress, size_t total, const char* message)
{
    if (atomic_load(&p_running))
    {
        print_error_prefix("THREAD");
        print(ERROR, MSG_NONE, "Progress monitor is already running");
        return false;
    }

    atomic_store(&p_running, true);
    p_progress = progress;
    p_message = message;
    p_total = total;

    pthread_create(&p_thread, NULL, progress_monitor_worker, NULL);
    if (!p_thread)
    {
        print_error_prefix("THREAD");
        print(ERROR, MSG_NONE, "Failed to create progress monitor thread");
        atomic_store(&p_running, false);
        return false;
    }

    return true;
}

void
progress_end(void)
{
    if (!atomic_load(&p_running))
    {
        print_error_prefix("THREAD");
        print(WARNING, MSG_NONE, "No progress monitor is running");
        return;
    }

    pthread_join(p_thread, NULL);
    atomic_store(&p_running, false);
}
