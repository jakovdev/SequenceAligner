#ifndef COMMON_H
#define COMMON_H

#include "user.h"
#include "macros.h"

#ifdef MODE_TUNE
#undef MODE_WRITE
#define MODE_WRITE 0
#endif

#define BLOSUM_SIZE (20)
#define ALIGN_BUF (MAX_SEQ_LEN * 2)

typedef struct {
    int matrix[BLOSUM_SIZE][BLOSUM_SIZE];
} ScoringMatrix;

typedef struct {
    char seq1_aligned[ALIGN_BUF];
    char seq2_aligned[ALIGN_BUF];
    int score;
    #if SIMILARITY_ANALYSIS == 1
    int matches;
    int mismatches;
    int gaps;
    double similarity;
    #endif
} Alignment;

typedef struct {
    #ifdef _WIN32
    HANDLE handle;
    DWORD bytes_written;
    #else
    int fd;
    #endif
    char buffer[WRITE_BUF];
    size_t pos;
} WriteBuffer;

INLINE void* mat_aligned_alloc(size_t alignment, size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void* ptr = NULL;
    posix_memalign(&ptr, alignment, size);
    return ptr;
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

#endif