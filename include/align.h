#ifndef ALIGN_H
#define ALIGN_H

#include "scoring.h"

#define STACK_MATRIX_THRESHOLD (128 * KiB)
#define MATRIX_SIZE(len1, len2) ((len1 + 1) * (len2 + 1))
#define MATRIX_BYTES(len1, len2) (MATRIX_SIZE(len1, len2) * sizeof(int))
#define MATRICES_3X_BYTES(len1, len2) (3 * MATRIX_BYTES(len1, len2))
#define USE_STACK_MATRIX(bytes) ((bytes) <= STACK_MATRIX_THRESHOLD)

// Direction enums for traceback
#define DIAG 0
#define UP 1
#define LEFT 2

static const int8_t next_i[] = {-1, -1, 0};    // DIAG, UP, LEFT
static const int8_t next_j[] = {-1, 0, -1};

INLINE void precompute_seq_indices(const char* restrict seq, int* restrict indices, size_t len) {
    #pragma GCC unroll 8
    for (size_t i = 0; i < len; ++i) {
        indices[i] = AMINO_LOOKUP[(int)seq[i]];
    }
}

#if MODE_CREATE_ALIGNED_STRINGS == 1
INLINE void construct_alignment_result(Alignment* restrict result, 
                                      char* restrict temp_seq1, 
                                      char* restrict temp_seq2, 
                                      int pos, 
                                      int score) {
    #pragma GCC unroll 8
    for (int k = 0; k < pos; k++) {
        result->seq1_aligned[k] = temp_seq1[pos - k - 1];
        result->seq2_aligned[k] = temp_seq2[pos - k - 1];
    }
    result->seq1_aligned[pos] = result->seq2_aligned[pos] = '\0';
    result->score = score;
}
#endif

INLINE int* allocate_matrix(int* stack_matrix, size_t bytes) {
    if (USE_STACK_MATRIX(bytes)) {
        return stack_matrix;
    } else {
        return (int*)huge_page_alloc(bytes);
    }
}

INLINE void free_matrix(int* matrix, int* stack_matrix) {
    if (matrix != stack_matrix) {
        aligned_free(matrix);
    }
}

#endif