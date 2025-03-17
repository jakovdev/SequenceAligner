#ifndef ALIGN_NW_H
#define ALIGN_NW_H

#include "align.h"

// Needleman-Wunsch global alignment with linear gap penalty
INLINE int nw_align(const char* seq1,
                    const size_t len1,
                    const char* seq2, 
                    const size_t len2,
                    const ScoringMatrix* restrict scoring) {
    size_t matrix_bytes = MATRIX_BYTES(len1, len2);
    int stack_matrix[USE_STACK_MATRIX(matrix_bytes) ? MATRIX_SIZE(len1, len2) : 1];
    int* restrict matrix = allocate_matrix(stack_matrix, matrix_bytes);
    const int cols = len1 + 1;
    const int GAP_PENALTY = g_gap_penalties.gap_penalty;

    int* restrict curr_row = matrix;
    curr_row[0] = 0;
    
    #ifdef USE_SIMD
    veci_t indices = FIRST_ROW_INDICES;
    for (int j = 1; j <= (int)len1; j += NUM_ELEMS) {
        veci_t values = mullo_epi32(indices, GAP_PENALTY_VEC);
        indices = add_epi32(indices, set1_epi32(NUM_ELEMS));
        storeu((veci_t*)&curr_row[j], values);
    }
    #else
    for(int j = 1; j <= (int)len1; j++) {
        matrix[j] = j * GAP_PENALTY;
    }
    #endif

    int seq1_indices[MAX_SEQ_LEN];
    precompute_seq_indices(seq1, seq1_indices, len1);

    // Fill matrix
    #pragma GCC unroll 8
    for (int i = 1; i <= (int)len2; ++i) {
        int* restrict prev_row = curr_row;
        curr_row = matrix + i * cols;
        curr_row[0] = i * GAP_PENALTY;
        int c2_idx = SEQUENCE_LOOKUP[(int)seq2[i - 1]];
        #pragma GCC unroll 4
        for (int j = 1; j <= (int)len1; j++) {
            int match = prev_row[j - 1] + scoring->matrix[seq1_indices[j - 1]][c2_idx];
            int del = prev_row[j] - GAP_PENALTY;
            int insert = curr_row[j - 1] - GAP_PENALTY;
            curr_row[j] = match > del ? (match > insert ? match : insert) : (del > insert ? del : insert);
        }
    }

    int score = matrix[len2 * cols + len1];

    free_matrix(matrix, stack_matrix);
    
    return score;
}

#endif