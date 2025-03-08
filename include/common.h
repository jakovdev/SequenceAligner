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

#define MAX_THREADS (32)

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


#endif