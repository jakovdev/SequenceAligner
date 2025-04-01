#ifndef COMMON_H
#define COMMON_H

#include "macros.h"

INLINE int
get_thread_count(void)
{
#ifdef _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors > MAX_THREADS ? MAX_THREADS : sysinfo.dwNumberOfProcessors;
#else
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    return nprocs > MAX_THREADS ? MAX_THREADS : nprocs;
#endif
}

INLINE double
get_time(void)
{
#ifdef _WIN32
    static double freq_inv = 0.0;
    if (freq_inv == 0.0)
    {
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);
        freq_inv = 1.0 / (double)freq.QuadPart;
    }

    LARGE_INTEGER count;
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * freq_inv;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

ALLOC INLINE void*
huge_page_alloc(size_t size)
{
    void* ptr = NULL;
#if USE_HUGE_PAGES && defined(__linux__)
    if (size >= HUGE_PAGE_THRESHOLD)
    {
        ptr = aligned_alloc(HUGE_PAGE_THRESHOLD, size);
        if (ptr)
        {
            madvise(ptr, size, MADV_HUGEPAGE);
        }
    }
#endif
    if (!ptr)
    {
        ptr = aligned_alloc(CACHE_LINE, size);
    }
    return ptr;
}

INLINE const char*
get_file_name(const char* path)
{
#ifdef _WIN32
    const char* name = strrchr(path, '\\');
#else
    const char* name = strrchr(path, '/');
#endif
    return name ? name + 1 : path;
}

INLINE size_t
get_available_memory(void)
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
                available_mem = strtoull(line + 13, NULL, 10) * KiB;
                break;
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

INLINE bool
check_matrix_exceeds_memory(size_t matrix_size, double safety_margin)
{
    size_t bytes_needed = matrix_size * matrix_size * sizeof(int);
    size_t safe_memory = (size_t)(get_available_memory() * safety_margin);
    return bytes_needed > safe_memory;
}

#endif