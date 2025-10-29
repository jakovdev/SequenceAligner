#include "system/arch.h"

// Language features
#if __STDC_VERSION__ >= 201112L || defined(_WIN32)
#define ALIGNED_ALLOC_AVAILABLE
#endif

#define HUGE_PAGE_THRESHOLD (2 * MiB)

#ifndef _WIN32
#include <stdio.h>
#include <sys/sysinfo.h>
#include <time.h>
#endif

#ifdef _WIN32
static double g_freq_inv;

void
time_init(void)
{
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    g_freq_inv = 1.0 / (double)freq.QuadPart;
}

double
time_current(void)
{
    LARGE_INTEGER count;
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * g_freq_inv;
}

#else

double
time_current(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#endif

void*
alloc_huge_page(size_t size)
{
    void* ptr = NULL;
#ifdef __linux__
    if (size >= HUGE_PAGE_THRESHOLD)
    {
        size_t aligned_size = (size + HUGE_PAGE_THRESHOLD - 1) & ~(HUGE_PAGE_THRESHOLD - 1);
#ifdef ALIGNED_ALLOC_AVAILABLE
        ptr = aligned_alloc(HUGE_PAGE_THRESHOLD, aligned_size);
#else
        if (posix_memalign(&ptr, HUGE_PAGE_THRESHOLD, aligned_size) != 0)
        {
            ptr = NULL;
        }

#endif
        if (ptr)
        {
            madvise(ptr, size, MADV_HUGEPAGE);
            return ptr;
        }
    }

#endif

    size_t aligned_size = (size + CACHE_LINE - 1) & ~(CACHE_LINE - 1);
#ifdef ALIGNED_ALLOC_AVAILABLE
    ptr = aligned_alloc(CACHE_LINE, aligned_size);
#else
    if (posix_memalign(&ptr, CACHE_LINE, aligned_size) != 0)
    {
        ptr = NULL;
    }

#endif

    return ptr;
}

size_t
available_memory(void)
{
    size_t available_mem = 0;

#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    available_mem = status.ullAvailPhys;
#else
    FILE* fp = fopen("/proc/meminfo", "r");
    if (fp)
    {
        char line[256];
        while (fgets(line, sizeof(line), fp))
        {
            if (strncmp(line, "MemAvailable:", 13) == 0)
            {
                char* endptr;
                unsigned long long val = strtoull(line + 13, &endptr, 10);
                if (endptr != line + 13)
                {
                    available_mem = val * KiB;
                    break;
                }
            }
        }

        fclose(fp);
    }

    if (available_mem == 0)
    {
        struct sysinfo info;
        if (sysinfo(&info) == 0)
        {
            available_mem = info.freeram * info.mem_unit;
        }
    }

#endif

    return available_mem;
}

const char*
file_name_path(const char* path)
{
#ifdef _WIN32
    const char* name = strrchr(path, '\\');
#else
    const char* name = strrchr(path, '/');
#endif
    return name ? name + 1 : path;
}
