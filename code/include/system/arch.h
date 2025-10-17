#pragma once
#ifndef SYSTEM_ARCH_H
#define SYSTEM_ARCH_H

// Language features
#if __STDC_VERSION__ >= 201112L || defined(_WIN32)
#define ALIGNED_ALLOC_AVAILABLE
#endif

// GCC/Clang specific macros
#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#define UNREACHABLE() __builtin_unreachable()
#define ALIGN __attribute__((aligned(CACHE_LINE)))
#define ALLOC __attribute__((malloc, alloc_size(1)))
#define UNUSED __attribute__((unused))
#define DESTRUCTOR __attribute__((destructor))
#define PRAGMA(n) _Pragma(#n)
#if defined(__clang__)
#define UNROLL(n) PRAGMA(unroll n)
#define VECTORIZE PRAGMA(clang loop vectorize(assume_safety))
#else // GCC
#define UNROLL(n) PRAGMA(GCC unroll n)
#define VECTORIZE PRAGMA(GCC ivdep)
#endif // clang
#elif defined(_MSC_VER)
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#define UNREACHABLE() __assume(0)
#define ALIGN __declspec(align(CACHE_LINE))
#define ALLOC
#define UNUSED
#define DESTRUCTOR
#define PRAGMA(n) __pragma(n)
#define UNROLL(n)
#define VECTORIZE PRAGMA(loop(ivdep))

#define strcasecmp _stricmp
#include <intrin.h>
#define __builtin_popcount(x) ((int)__popcnt(x))
#define __builtin_popcountll(x) ((int)__popcnt64(x))

static inline int
__builtin_ctz(unsigned int x)
{
    unsigned long index;
    _BitScanForward(&index, x);
    return (int)index;
}

static inline int
__builtin_ctzll(unsigned long long x)
{
    unsigned long index;
    _BitScanForward64(&index, x);
    return (int)index;
}

#endif

// System and memory-related constants
#define KiB (1ULL << 10)
#define MiB (KiB << 10)
#define GiB (MiB << 10)

#define HUGE_PAGE_THRESHOLD (2 * MiB)

#define CACHE_LINE ((size_t)64)

#define ALIGN_POW2(value, pow2) (((value) + ((pow2 >> 1) - 1)) / (pow2)) * (pow2)

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>
#if __has_include(<Shlwapi.h>)
#include <Shlwapi.h>
#else // MinGW
#include <shlwapi.h>
#endif

#ifdef ERROR
#undef ERROR
#endif

typedef HANDLE pthread_t;
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

#define PIN_THREAD(thread_id) SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << thread_id)

#define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)

#define strcasestr(haystack, needle) StrStrIA(haystack, needle)
#define usleep(microseconds) Sleep((microseconds) / 1000)

#define MAX max
#define MIN min

#else // POSIX/Linux

#define MAX_PATH (260)

#include <alloca.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
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

#define aligned_free(ptr) free(ptr)

#endif

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
#define restrict __restrict
#define CAST(ptr) static_cast<decltype(ptr)>
#else
#define CAST(ptr)
#endif

#define ALLOCATION(ptr, count, func) CAST(ptr)(func((count) * sizeof(*(ptr))))

#define MALLOC(ptr, count) ALLOCATION(ptr, count, malloc)
#define ALLOCA(ptr, count) ALLOCATION(ptr, count, alloca)
#define REALLOC(ptr, count) CAST(ptr)(realloc(ptr, (count) * sizeof(*(ptr))))

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
#define ctz(x) ((num_t)__builtin_ctzll(x))
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
#define ctz(x) ((num_t)__builtin_ctz(x))
#define loadu _mm256_loadu_si256
#define storeu _mm256_storeu_si256
#define add_epi32 _mm256_add_epi32
#define sub_epi32 _mm256_sub_epi32
#define mullo_epi32 _mm256_mullo_epi32
#define set1_epi32 _mm256_set1_epi32
#define set1_epi8 _mm256_set1_epi8
#define cmpeq_epi8 _mm256_cmpeq_epi8
#define movemask_epi8(x) ((num_t)_mm256_movemask_epi8(x))
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
#define ctz(x) ((num_t)__builtin_ctz(x))
#define loadu _mm_loadu_si128
#define storeu _mm_storeu_si128
#define add_epi32 _mm_add_epi32
#define sub_epi32 _mm_sub_epi32
#define mullo_epi32 _mm_mullo_epi32_fallback
#define set1_epi32 _mm_set1_epi32
#define set1_epi8 _mm_set1_epi8
#define cmpeq_epi8 _mm_cmpeq_epi8
#define movemask_epi8(x) ((num_t)_mm_movemask_epi8(x))
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

static inline long
thread_count(void)
{
#ifdef _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return (long)sysinfo.dwNumberOfProcessors;
#else
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    return nprocs;
#endif
}

static inline double
time_current(void)
{
#ifdef _WIN32
    // TODO: Check

    // static double freq_inv = 0.0;
    // if (freq_inv == 0.0)
    // {
    //     LARGE_INTEGER freq;
    //     QueryPerformanceFrequency(&freq);
    //     freq_inv = 1.0 / (double)freq.QuadPart;
    // }

    // LARGE_INTEGER count;
    // QueryPerformanceCounter(&count);
    // return (double)count.QuadPart * freq_inv;

    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

ALLOC static inline void*
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

static inline const char*
file_name_path(const char* path)
{
#ifdef _WIN32
    const char* name = strrchr(path, '\\');
#else
    const char* name = strrchr(path, '/');
#endif
    return name ? name + 1 : path;
}

#endif // SYSTEM_ARCH_H