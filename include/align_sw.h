#ifndef ALIGN_SW_H
#define ALIGN_SW_H

#include "align.h"

// Smith-Waterman local alignment with affine gap penalty
INLINE int sw_align(const char* seq1,
                    const size_t len1,
                    const char* seq2, 
                    const size_t len2,
                    const ScoringMatrix* restrict scoring) {
    // Three matrices needed for affine gap penalties: match, gap_x, gap_y
    size_t matrices_bytes = MATRICES_3X_BYTES(len1, len2);
    int stack_matrix[USE_STACK_MATRIX(matrices_bytes) ? 3 * MATRIX_SIZE(len1, len2) : 1];
    int* restrict matrix = allocate_matrix(stack_matrix, matrices_bytes);
    const int GAP_START = g_gap_penalties.gap_open;
    const int GAP_EXTEND = g_gap_penalties.gap_extend;

    int* restrict match = matrix;
    int* restrict gap_x = matrix + MATRIX_SIZE(len1, len2);
    int* restrict gap_y = matrix + 2 * MATRIX_SIZE(len1, len2);
    const int cols = len1 + 1;

    #ifdef USE_SIMD
    veci_t zero_vec = setzero_si();
    veci_t int_min_half = set1_epi32(INT_MIN/2);
    
    for (int j = 0; j <= (int)len1; j += NUM_ELEMS) {
        int remaining = (int)len1 + 1 - j;
        if (remaining >= NUM_ELEMS) {
            storeu((veci_t*)&match[j], zero_vec);
            storeu((veci_t*)&gap_x[j], int_min_half);
            storeu((veci_t*)&gap_y[j], int_min_half);
        } else {
            for (int k = 0; k < remaining; k++) {
                match[j+k] = 0;
                gap_x[j+k] = gap_y[j+k] = INT_MIN/2;
            }
        }
    }
    #else
    match[0] = 0;
    gap_x[0] = gap_y[0] = INT_MIN/2;
    
    #pragma GCC unroll 8
    for (int j = 1; j <= (int)len1; j++) {
        match[j] = 0;
        gap_x[j] = gap_y[j] = INT_MIN/2;
    }
    #endif
    
    #pragma GCC unroll 8
    for (int i = 1; i <= (int)len2; i++) {
        int idx = i * cols;
        match[idx] = 0;
        gap_x[idx] = gap_y[idx] = INT_MIN/2;
    }

    int seq1_indices[MAX_SEQ_LEN];
    precompute_seq_indices(seq1, seq1_indices, len1);

    int final_score = 0;
    int max_i = 0, max_j = 0;

    // Fill matrices
    for (int i = 1; i <= (int)len2; ++i) {
        int row_offset = i * cols;
        int prev_row_offset = (i-1) * cols;
        int c2_idx = SEQUENCE_LOOKUP[(int)seq2[i - 1]];

        PREFETCH(&match[row_offset + PREFETCH_DISTANCE]);
        PREFETCH(&gap_x[row_offset + PREFETCH_DISTANCE]);
        PREFETCH(&gap_y[row_offset + PREFETCH_DISTANCE]);
        
        for (int j = 1; j <= (int)len1; j++) {
            // Calculate match score (diagonal move)
            int similarity = scoring->matrix[seq1_indices[j - 1]][c2_idx];
            int diag_score = match[prev_row_offset + j - 1] + similarity;
            
            // Calculate gap scores
            int prev_match_x = match[row_offset + j - 1];
            int prev_gap_x = gap_x[row_offset + j - 1];
            int prev_match_y = match[prev_row_offset + j];
            int prev_gap_y = gap_y[prev_row_offset + j];
            
            int open_x = prev_match_x - GAP_START;
            int extend_x = prev_gap_x - GAP_EXTEND;
            int open_y = prev_match_y - GAP_START;
            int extend_y = prev_gap_y - GAP_EXTEND;
            
            gap_x[row_offset + j] = (open_x > extend_x) ? open_x : extend_x;
            gap_y[row_offset + j] = (open_y > extend_y) ? open_y : extend_y;
            
            // Find best score (0 is minimum for local alignment)
            int curr_gap_x = gap_x[row_offset + j];
            int curr_gap_y = gap_y[row_offset + j];
            
            int best = 0;
            best = diag_score > best ? diag_score : best;
            best = curr_gap_x > best ? curr_gap_x : best;
            best = curr_gap_y > best ? curr_gap_y : best;
            
            match[row_offset + j] = best;
            
            // Update max score if needed
            int is_new_max = best > final_score;
            max_i = is_new_max * i + (1 - is_new_max) * max_i;
            max_j = is_new_max * j + (1 - is_new_max) * max_j;
            final_score = is_new_max * best + (1 - is_new_max) * final_score;
        }
    }

    free_matrix(matrix, stack_matrix);
    
    return final_score;
}

#endif