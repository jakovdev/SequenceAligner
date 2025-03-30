#ifndef ALIGN_GA_H
#define ALIGN_GA_H

#include "align.h"

// Gotoh global alignment with affine gap penalty
INLINE int
ga_align(const char* seq1,
         const size_t len1,
         const char* seq2,
         const size_t len2,
         const ScoringMatrix* restrict scoring)
{
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

    match[0] = 0;
    gap_x[0] = gap_y[0] = INT_MIN / 2; // So that it's not chosen / no overflow

#ifdef USE_SIMD
    veci_t indices = FIRST_ROW_INDICES;
    veci_t int_min_half = set1_epi32(INT_MIN / 2);

    for (int j = 1; j <= (int)len1; j += NUM_ELEMS)
    {
        int remaining = (int)len1 + 1 - j;
        if (remaining >= NUM_ELEMS)
        {
            veci_t position_indices = add_epi32(indices, set1_epi32(j - 1));
            veci_t values = add_epi32(mullo_epi32(position_indices, GAP_EXTEND_VEC), GAP_START_VEC);
            values = sub_epi32(setzero_si(), values);

            storeu((veci_t*)&match[j], values);
            storeu((veci_t*)&gap_x[j], values);
            storeu((veci_t*)&gap_y[j], int_min_half);
        }
        else
        {
            // Handle remaining elements
            for (int k = 0; k < remaining; k++)
            {
                match[j + k] = -(GAP_START + (j + k - 1) * GAP_EXTEND);
                gap_x[j + k] = match[j + k];
                gap_y[j + k] = INT_MIN / 2;
            }
        }
        indices = add_epi32(indices, set1_epi32(NUM_ELEMS));
    }
#else
#pragma GCC unroll 8
    for (int j = 1; j <= (int)len1; j++)
    {
        match[j] = -(GAP_START + (j - 1) * GAP_EXTEND);
        gap_x[j] = match[j];
        gap_y[j] = INT_MIN / 2;
    }
#endif

#pragma GCC unroll 8
    for (int i = 1; i <= (int)len2; i++)
    {
        int idx = i * cols;
        match[idx] = -(GAP_START + (i - 1) * GAP_EXTEND);
        gap_y[idx] = match[idx];
        gap_x[idx] = INT_MIN / 2;
    }

    SeqIndices seq1_indices = { 0 };
    precompute_seq_indices(&seq1_indices, seq1, len1);

    // Fill matrices
    for (int i = 1; i <= (int)len2; ++i)
    {
        int row_offset = i * cols;
        int prev_row_offset = (i - 1) * cols;
        int c2_idx = SEQUENCE_LOOKUP[(unsigned char)seq2[i - 1]];

        // Prefetch the next row for better cache behavior
        PREFETCH(&match[row_offset + PREFETCH_DISTANCE]);
        PREFETCH(&gap_x[row_offset + PREFETCH_DISTANCE]);
        PREFETCH(&gap_y[row_offset + PREFETCH_DISTANCE]);

        for (int j = 1; j <= (int)len1; j++)
        {
            // Calculate match score (diagonal move)
            int similarity = scoring->matrix[seq1_indices.data[j - 1]][c2_idx];
            int diag_score = match[prev_row_offset + j - 1] + similarity;

            // Calculate gap in X score (horizontal move)
            int open_x = match[row_offset + j - 1] - GAP_START;
            int extend_x = gap_x[row_offset + j - 1] - GAP_EXTEND;
            gap_x[row_offset + j] = (open_x > extend_x) ? open_x : extend_x;

            // Calculate gap in Y score (vertical move)
            int open_y = match[prev_row_offset + j] - GAP_START;
            int extend_y = gap_y[prev_row_offset + j] - GAP_EXTEND;
            gap_y[row_offset + j] = (open_y > extend_y) ? open_y : extend_y;

            // Find the best score among the three options
            int best = diag_score;
            if (gap_x[row_offset + j] > best)
            {
                best = gap_x[row_offset + j];
            }
            if (gap_y[row_offset + j] > best)
            {
                best = gap_y[row_offset + j];
            }

            match[row_offset + j] = best;
        }
    }

    int score = match[len2 * cols + len1];

    free_seq_indices(&seq1_indices);
    free_matrix(matrix, stack_matrix);
    return score;
}

#endif