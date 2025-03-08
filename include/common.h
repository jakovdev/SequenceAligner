#ifndef COMMON_H
#define COMMON_H

#include "user.h"
#include "macros.h"

#define BLOSUM_SIZE (20)
#if MODE_CREATE_ALIGNED_STRINGS == 1
#define ALIGN_BUF (MAX_SEQ_LEN * 2)
#endif

typedef struct {
    int matrix[BLOSUM_SIZE][BLOSUM_SIZE];
} ScoringMatrix;

typedef struct {
    #if MODE_CREATE_ALIGNED_STRINGS == 1
    char seq1_aligned[ALIGN_BUF];
    char seq2_aligned[ALIGN_BUF];
    #endif
    int score;
} Alignment;

typedef struct {
    char data[MAX_SEQ_LEN];
} Sequence;

INLINE int get_thread_count(void) {
    #ifdef _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors > MAX_THREADS ? MAX_THREADS : sysinfo.dwNumberOfProcessors;
    #else
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    return nprocs > MAX_THREADS ? MAX_THREADS : nprocs;
    #endif
}

INLINE double get_time(void) {
    #ifdef _WIN32
    static double freq_inv = 0.0;
    if (freq_inv == 0.0) {
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

INLINE void* huge_page_alloc(size_t size) {
    void* ptr = NULL;
    #if USE_HUGE_PAGES && defined(__linux__)
    if (size >= HUGE_PAGE_THRESHOLD) {
        ptr = aligned_alloc(HUGE_PAGE_THRESHOLD, size);
        if (ptr) {
            #ifdef MADV_HUGEPAGE
            madvise(ptr, size, MADV_HUGEPAGE);
            #endif
        }
    }
    #endif
    if (!ptr) {
        ptr = aligned_alloc(CACHE_LINE, size);
    }
    return ptr;
}


#endif