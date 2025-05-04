#ifndef ARCH_H
#define ARCH_H

// GCC/Clang specific macros
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#define UNREACHABLE() __builtin_unreachable()
#define ALIGN __attribute__((aligned(CACHE_LINE)))
#define ALLOC __attribute__((malloc, alloc_size(1)))
#define UNUSED __attribute__((unused))
#define DESTRUCTOR __attribute__((destructor))
#define CLEANUP(function) __attribute__((cleanup(function)))
#define PRAGMA(n) _Pragma(#n)
#if defined(__clang__)
#define UNROLL(n) PRAGMA(unroll n)
#else
#define UNROLL(n) PRAGMA(GCC unroll n)
#endif

// System and memory-related constants
#define KiB (1ULL << 10)
#define MiB (KiB << 10)
#define GiB (MiB << 10)

#define HUGE_PAGE_THRESHOLD (2 * MiB)

#define CACHE_LINE 64

#ifdef __cplusplus
#define restrict __restrict
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32

#include <Shlwapi.h>
#include <conio.h>
#include <io.h>
#include <malloc.h>
#include <synchapi.h>
#include <windows.h>
#include <winioctl.h>

#ifdef ERROR
#undef ERROR
#endif

#ifdef OPTIONAL
#undef OPTIONAL
#endif

typedef HANDLE pthread_t;
typedef HANDLE sem_t;
typedef HANDLE pthread_mutex_t;

#define T_Func DWORD WINAPI
#define T_Ret(x) return (DWORD)(size_t)(x)

#define pthread_create(threads, _, function, arg)                                                  \
    (void)(*threads = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)function, arg, 0, NULL))
#define pthread_mutex_lock(mutex) WaitForSingleObject(mutex, INFINITE)
#define pthread_mutex_unlock(mutex) ReleaseMutex(mutex)
#define PTHREAD_MUTEX_INITIALIZER CreateMutex(NULL, FALSE, NULL)
#define pthread_mutex_destroy(mutex) CloseHandle(mutex)

#define pthread_join(thread_id, _) WaitForSingleObject(thread_id, INFINITE)
#define sem_init(sem, _, value) *sem = CreateSemaphore(NULL, value, LONG_MAX, NULL)
#define sem_post(sem) ReleaseSemaphore(*sem, 1, NULL)
#define sem_wait(sem) WaitForSingleObject(*sem, INFINITE)
#define sem_destroy(sem) CloseHandle(*sem)

#define PIN_THREAD(thread_id) SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << thread_id)
#define SET_HIGH_CLASS() SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS)

#define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)

#define strcasestr(haystack, needle) StrStrIA(haystack, needle)
#define usleep(microseconds) Sleep((microseconds) / 1000)

#define MAX max
#define MIN min

#else // POSIX/Linux

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef __USE_GNU
#define __USE_GNU
#endif
#define MAX_PATH (260)

#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/types.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#define T_Func void*
#define T_Ret(x) return (x)

#define PIN_THREAD(thread_id)                                                                      \
    do                                                                                             \
    {                                                                                              \
        cpu_set_t cpuset;                                                                          \
        CPU_ZERO(&cpuset);                                                                         \
        CPU_SET(thread_id, &cpuset);                                                               \
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);                        \
    } while (false)

#define SET_HIGH_CLASS()

#define aligned_free(ptr) free(ptr)

#endif

// SIMD detection and intrinsics
#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>
#include <x86intrin.h>
#define USE_SIMD
#define USE_AVX512
typedef __m512i veci_t;
typedef __mmask64 num_t;
#define BYTES (64)
#define NUM_ELEMS (16)
#define ctz __builtin_ctzll
#define loadu _mm512_loadu_si512
#define storeu _mm512_storeu_si512
#define add_epi32 _mm512_add_epi32
#define sub_epi32 _mm512_sub_epi32
#define mullo_epi32 _mm512_mullo_epi32
#define set1_epi32 _mm512_set1_epi32
#define set1_epi8 _mm512_set1_epi8
#define cmpeq_epi8 _mm512_cmpeq_epi8_mask
#define movemask_epi8
#define or_mask(a, b) ((a) | (b))
#define or_si _mm512_or_si512
#define setzero_si _mm512_setzero_si512
#define and_si _mm512_and_si512
#define setr_epi32 _mm512_setr_epi32
#define set_row_indices() _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)

#elif defined(__AVX2__)
#include <immintrin.h>
#define USE_SIMD
#define USE_AVX2
typedef __m256i veci_t;
typedef uint32_t num_t;
#define BYTES (32)
#define NUM_ELEMS (8)
#define ctz __builtin_ctz
#define loadu _mm256_loadu_si256
#define storeu _mm256_storeu_si256
#define add_epi32 _mm256_add_epi32
#define sub_epi32 _mm256_sub_epi32
#define mullo_epi32 _mm256_mullo_epi32
#define set1_epi32 _mm256_set1_epi32
#define set1_epi8 _mm256_set1_epi8
#define cmpeq_epi8 _mm256_cmpeq_epi8
#define movemask_epi8 _mm256_movemask_epi8
#define or_si _mm256_or_si256
#define setzero_si _mm256_setzero_si256
#define and_si _mm256_and_si256
#define setr_epi32 _mm256_setr_epi32
#define set_row_indices() _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8)

#elif defined(__SSE2__)
#include <emmintrin.h>
#define USE_SIMD
#define USE_SSE
typedef __m128i veci_t;
typedef uint16_t num_t;
#define BYTES (16)
#define NUM_ELEMS (4)
#define ctz __builtin_ctz
#define loadu _mm_loadu_si128
#define storeu _mm_storeu_si128
#define add_epi32 _mm_add_epi32
#define sub_epi32 _mm_sub_epi32
#define mullo_epi32 _mm_mullo_epi32_fallback
#define set1_epi32 _mm_set1_epi32
#define set1_epi8 _mm_set1_epi8
#define cmpeq_epi8 _mm_cmpeq_epi8
#define movemask_epi8 _mm_movemask_epi8
#define or_si _mm_or_si128
#define setzero_si _mm_setzero_si128
#define and_si _mm_and_si128
#define setr_epi32 _mm_setr_epi32
#define set_row_indices() _mm_setr_epi32(1, 2, 3, 4)

static inline __m128i
_mm_mullo_epi32_fallback(__m128i a, __m128i b)
{
    __m128i tmp1 = _mm_mul_epu32(a, b);
    __m128i tmp2 = _mm_mul_epu32(_mm_srli_si128(a, 4), _mm_srli_si128(b, 4));
    return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0, 0, 2, 0)),
                              _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0, 0, 2, 0)));
}

#else
#define prefetch(x)
#define prefetch_write(x)
#endif

#ifdef USE_SIMD
#define PREFETCH_DISTANCE (BYTES << 4)
#define prefetch(x) _mm_prefetch((const char*)(x), _MM_HINT_T0)
#define prefetch_write(x) _mm_prefetch((const char*)(x), _MM_HINT_T1)
#endif

static inline int
thread_count(void)
{
#ifdef _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
#else
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    return nprocs;
#endif
}

static inline double
time_current(void)
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

ALLOC static inline void*
alloc_huge_page(size_t size)
{
    void* ptr = NULL;
#ifdef __linux__
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

static inline size_t
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

#endif // ARCH_H