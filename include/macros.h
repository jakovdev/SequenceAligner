#ifndef MACROS_H
#define MACROS_H

#ifdef _WIN32

#include <windows.h>
#include <io.h>
#include <winioctl.h>
#include <malloc.h>

#ifdef CROSS_COMPILE
#include <shlwapi.h>
#else
#include <Shlwapi.h>
#endif

typedef HANDLE pthread_t;

#define T_Func DWORD WINAPI
#define T_Ret(x) return (DWORD)(size_t)(x)

#define PIN_THREAD(t_id) SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << t_id)
#define SET_HIGH_CLASS() SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS)

#define mat_aligned_free(p) _aligned_free(p)

#undef max
#undef min

#else

#define _GNU_SOURCE
#define MAX_PATH (260) 

#include <time.h>
#include <unistd.h>
#include <sched.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <pthread.h>

#define T_Func void*
#define T_Ret(x) return (x)

#define PIN_THREAD(t_id) do { \
    cpu_set_t cpuset; \
    CPU_ZERO(&cpuset); \
    CPU_SET(t_id, &cpuset); \
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset); \
} while(0)
#define SET_HIGH_CLASS() nice(-20)    // requires root I think

#define mat_aligned_free(p) free(p)

#endif

#include <fcntl.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
#include <string.h>
#include <stdbool.h>
#include <limits.h>

#define max(a, b) ((a) + (((b) - (a)) & ((b) - (a)) >> 31))
#define min(a, b) ((a) - (((a) - (b)) & ((a) - (b)) >> 31))

#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#define PREFETCH(x) _mm_prefetch((const char*)(x), _MM_HINT_T0)

#define INLINE static inline
#define CACHE_LINE (64)
#define ALIGN __attribute__((aligned(CACHE_LINE)))

// Will fix later...
/*#if defined(__AVX512F__)
    #define USE_AVX
    typedef __m512i veci_t;
    typedef uint64_t num_t;
    #define BYTES (64)
    #define NUM_ELEMS (16)
    #define ctz __builtin_ctzll
    #define loadu _mm512_loadu_si512
    #define storeu _mm512_storeu_si512
    #define add_epi32 _mm512_add_epi32
    #define mullo_epi32 _mm512_mullo_epi32
    #define set1_epi32 _mm512_set1_epi32
    #define set1_epi8 _mm512_set1_epi8
    #define cmpeq_epi8 _mm512_cmpeq_epi8
    #define movemask_epi8 _mm512_movm_epi8
    #define or_si _mm512_or_si512
    #define setzero_si _mm512_setzero_si512
    #define and_si _mm512_and_si512
    #define setr_indicies _mm512_setr_epi32(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16)*/
#if defined(__AVX2__)
    #define USE_AVX
    typedef __m256i veci_t;
    typedef uint32_t num_t;
    #define BYTES (32)
    #define NUM_ELEMS (8)
    #define ctz __builtin_ctz
    #define loadu _mm256_loadu_si256
    #define storeu _mm256_storeu_si256
    #define add_epi32 _mm256_add_epi32
    #define mullo_epi32 _mm256_mullo_epi32
    #define set1_epi32 _mm256_set1_epi32
    #define set1_epi8 _mm256_set1_epi8
    #define cmpeq_epi8 _mm256_cmpeq_epi8
    #define movemask_epi8 _mm256_movemask_epi8
    #define or_si _mm256_or_si256
    #define setzero_si _mm256_setzero_si256
    #define and_si _mm256_and_si256
    #define setr_indicies _mm256_setr_epi32(1,2,3,4,5,6,7,8)
#elif defined(__SSE2__)
    #define USE_AVX
    typedef __m128i veci_t;
    typedef uint16_t num_t;
    #define BYTES (16)
    #define NUM_ELEMS (4)
    #define ctz __builtin_ctz
    #define loadu _mm_loadu_si128
    #define storeu _mm_storeu_si128
    #define add_epi32 _mm_add_epi32
    #define mullo_epi32 _mm_mullo_epi32
    #define set1_epi32 _mm_set1_epi32
    #define set1_epi8 _mm_set1_epi8
    #define cmpeq_epi8 _mm_cmpeq_epi8
    #define movemask_epi8 _mm_movemask_epi8
    #define or_si _mm_or_si128
    #define setzero_si _mm_setzero_si128
    #define and_si _mm_and_si128
    #define setr_indicies _mm_setr_epi32(1,2,3,4)
#endif

#endif