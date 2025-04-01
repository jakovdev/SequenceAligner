#ifndef SEQALIGN_H
#define SEQALIGN_H

#include "args.h"
#include "methods.h"
#include "scoring.h"

#define STACK_MATRIX_THRESHOLD (128 * KiB)
#define MATRIX_SIZE(len1, len2) ((len1 + 1) * (len2 + 1))
#define MATRIX_BYTES(len1, len2) (MATRIX_SIZE(len1, len2) * sizeof(int))
#define MATRICES_3X_BYTES(len1, len2) (3 * MATRIX_BYTES(len1, len2))
#define USE_STACK_MATRIX(bytes) ((bytes) <= STACK_MATRIX_THRESHOLD)

typedef struct
{
    int* data;
    size_t size;
    bool is_stack;
} SeqIndices;

INLINE void
seq_indices_precompute(SeqIndices* indices, const char* restrict seq, size_t len)
{
    indices->size = len;

    if (len <= MAX_STACK_SEQUENCE_LENGTH)
    {
        int* stack_buffer = alloca(len * sizeof(int));
        indices->data = stack_buffer;
        indices->is_stack = true;
    }
    else
    {
        indices->data = malloc(len * sizeof(*indices->data));
        indices->is_stack = false;
    }

#ifdef USE_SIMD
    size_t vector_len = (len / BYTES) * BYTES;
    size_t i = 0;

    for (; i < vector_len; i += BYTES)
    {
        prefetch(seq + i + BYTES * 2);

        for (size_t j = 0; j < BYTES; j++)
        {
            indices->data[i + j] = SEQUENCE_LOOKUP[(unsigned char)seq[i + j]];
        }
    }

    for (; i < len; i++)
    {
        indices->data[i] = SEQUENCE_LOOKUP[(unsigned char)seq[i]];
    }
#else
#pragma GCC unroll 8
    for (size_t i = 0; i < len; ++i)
    {
        indices->data[i] = SEQUENCE_LOOKUP[(unsigned char)seq[i]];
    }
#endif
}

INLINE void
seq_indices_free(SeqIndices* indices)
{
    if (!indices->is_stack && indices->data)
    {
        free(indices->data);
        indices->data = NULL;
    }
}

INLINE int*
matrix_alloc(int* stack_matrix, size_t bytes)
{
    if (USE_STACK_MATRIX(bytes))
    {
        return stack_matrix;
    }
    else
    {
        return (int*)alloc_huge_page(bytes);
    }
}

INLINE void
matrix_free(int* matrix, int* stack_matrix)
{
    if (matrix != stack_matrix)
    {
        aligned_free(matrix);
    }
}

#define INIT_LINEAR_GLOBAL(matrix, cols, gap_penalty)                                              \
    do                                                                                             \
    {                                                                                              \
        matrix[0] = 0;                                                                             \
                                                                                                   \
        _Pragma("GCC unroll 8") for (int j = 1; j <= (int)len1; j++)                               \
        {                                                                                          \
            matrix[j] = j * (gap_penalty);                                                         \
        }                                                                                          \
                                                                                                   \
        _Pragma("GCC unroll 8") for (int i = 1; i <= (int)len2; i++)                               \
        {                                                                                          \
            matrix[i * cols] = i * (gap_penalty);                                                  \
        }                                                                                          \
    } while (0)

#define INIT_AFFINE_GLOBAL(match, gap_x, gap_y, cols, gap_start, gap_extend)                       \
    do                                                                                             \
    {                                                                                              \
        match[0] = 0;                                                                              \
        gap_x[0] = gap_y[0] = INT_MIN / 2;                                                         \
                                                                                                   \
        _Pragma("GCC unroll 8") for (int j = 1; j <= (int)len1; j++)                               \
        {                                                                                          \
            match[j] = -(gap_start + (j - 1) * gap_extend);                                        \
            gap_x[j] = match[j];                                                                   \
            gap_y[j] = INT_MIN / 2;                                                                \
        }                                                                                          \
                                                                                                   \
        _Pragma("GCC unroll 8") for (int i = 1; i <= (int)len2; i++)                               \
        {                                                                                          \
            int idx = i * cols;                                                                    \
            match[idx] = -(gap_start + (i - 1) * gap_extend);                                      \
            gap_y[idx] = match[idx];                                                               \
            gap_x[idx] = INT_MIN / 2;                                                              \
        }                                                                                          \
    } while (0)

#define INIT_AFFINE_LOCAL(match, gap_x, gap_y, cols)                                               \
    do                                                                                             \
    {                                                                                              \
        match[0] = 0;                                                                              \
        gap_x[0] = gap_y[0] = INT_MIN / 2;                                                         \
                                                                                                   \
        _Pragma("GCC unroll 8") for (int j = 1; j <= (int)len1; j++)                               \
        {                                                                                          \
            match[j] = 0;                                                                          \
            gap_x[j] = gap_y[j] = INT_MIN / 2;                                                     \
        }                                                                                          \
                                                                                                   \
        _Pragma("GCC unroll 8") for (int i = 1; i <= (int)len2; i++)                               \
        {                                                                                          \
            int idx = i * cols;                                                                    \
            match[idx] = 0;                                                                        \
            gap_x[idx] = gap_y[idx] = INT_MIN / 2;                                                 \
        }                                                                                          \
    } while (0)

#define FILL_LINEAR_GLOBAL(matrix, cols, gap_penalty, similarity)                                  \
    do                                                                                             \
    {                                                                                              \
        for (int i = 1; i <= (int)len2; ++i)                                                       \
        {                                                                                          \
            int row_offset = i * cols;                                                             \
            int prev_row_offset = (i - 1) * cols;                                                  \
            int c2_idx = SEQUENCE_LOOKUP[(unsigned char)seq2[i - 1]];                              \
                                                                                                   \
            prefetch(&matrix[row_offset + PREFETCH_DISTANCE]);                                     \
                                                                                                   \
            _Pragma("GCC unroll 4") for (int j = 1; j <= (int)len1; j++)                           \
            {                                                                                      \
                int match = matrix[prev_row_offset + j - 1] +                                      \
                            scoring->matrix[seq1_indices.data[j - 1]][c2_idx];                     \
                int del = matrix[prev_row_offset + j] - (gap_penalty);                             \
                int insert = matrix[row_offset + j - 1] - (gap_penalty);                           \
                matrix[row_offset + j] = match > del ? (match > insert ? match : insert)           \
                                                     : (del > insert ? del : insert);              \
            }                                                                                      \
        }                                                                                          \
    } while (0)

#define FILL_AFFINE_GLOBAL(match, gap_x, gap_y, cols, gap_start, gap_extend)                       \
    do                                                                                             \
    {                                                                                              \
        for (int i = 1; i <= (int)len2; ++i)                                                       \
        {                                                                                          \
            int row_offset = i * cols;                                                             \
            int prev_row_offset = (i - 1) * cols;                                                  \
            int c2_idx = SEQUENCE_LOOKUP[(unsigned char)seq2[i - 1]];                              \
                                                                                                   \
            prefetch(&match[row_offset + PREFETCH_DISTANCE]);                                      \
            prefetch(&gap_x[row_offset + PREFETCH_DISTANCE]);                                      \
            prefetch(&gap_y[row_offset + PREFETCH_DISTANCE]);                                      \
                                                                                                   \
            for (int j = 1; j <= (int)len1; j++)                                                   \
            {                                                                                      \
                int similarity = scoring->matrix[seq1_indices.data[j - 1]][c2_idx];                \
                int diag_score = match[prev_row_offset + j - 1] + similarity;                      \
                                                                                                   \
                int open_x = match[row_offset + j - 1] - (gap_start);                              \
                int extend_x = gap_x[row_offset + j - 1] - (gap_extend);                           \
                gap_x[row_offset + j] = (open_x > extend_x) ? open_x : extend_x;                   \
                                                                                                   \
                int open_y = match[prev_row_offset + j] - (gap_start);                             \
                int extend_y = gap_y[prev_row_offset + j] - (gap_extend);                          \
                gap_y[row_offset + j] = (open_y > extend_y) ? open_y : extend_y;                   \
                                                                                                   \
                int best = diag_score;                                                             \
                if (gap_x[row_offset + j] > best)                                                  \
                    best = gap_x[row_offset + j];                                                  \
                if (gap_y[row_offset + j] > best)                                                  \
                    best = gap_y[row_offset + j];                                                  \
                                                                                                   \
                match[row_offset + j] = best;                                                      \
            }                                                                                      \
        }                                                                                          \
    } while (0)

#define FILL_AFFINE_LOCAL(match,                                                                   \
                          gap_x,                                                                   \
                          gap_y,                                                                   \
                          cols,                                                                    \
                          gap_start,                                                               \
                          gap_extend,                                                              \
                          final_score,                                                             \
                          max_i,                                                                   \
                          max_j)                                                                   \
    do                                                                                             \
    {                                                                                              \
        final_score = 0;                                                                           \
        max_i = max_j = 0;                                                                         \
                                                                                                   \
        for (int i = 1; i <= (int)len2; ++i)                                                       \
        {                                                                                          \
            int row_offset = i * cols;                                                             \
            int prev_row_offset = (i - 1) * cols;                                                  \
            int c2_idx = SEQUENCE_LOOKUP[(unsigned char)seq2[i - 1]];                              \
                                                                                                   \
            prefetch(&match[row_offset + PREFETCH_DISTANCE]);                                      \
            prefetch(&gap_x[row_offset + PREFETCH_DISTANCE]);                                      \
            prefetch(&gap_y[row_offset + PREFETCH_DISTANCE]);                                      \
                                                                                                   \
            for (int j = 1; j <= (int)len1; j++)                                                   \
            {                                                                                      \
                int similarity = scoring->matrix[seq1_indices.data[j - 1]][c2_idx];                \
                int diag_score = match[prev_row_offset + j - 1] + similarity;                      \
                                                                                                   \
                int prev_match_x = match[row_offset + j - 1];                                      \
                int prev_gap_x = gap_x[row_offset + j - 1];                                        \
                int prev_match_y = match[prev_row_offset + j];                                     \
                int prev_gap_y = gap_y[prev_row_offset + j];                                       \
                                                                                                   \
                int open_x = prev_match_x - (gap_start);                                           \
                int extend_x = prev_gap_x - (gap_extend);                                          \
                int open_y = prev_match_y - (gap_start);                                           \
                int extend_y = prev_gap_y - (gap_extend);                                          \
                                                                                                   \
                gap_x[row_offset + j] = (open_x > extend_x) ? open_x : extend_x;                   \
                gap_y[row_offset + j] = (open_y > extend_y) ? open_y : extend_y;                   \
                                                                                                   \
                int curr_gap_x = gap_x[row_offset + j];                                            \
                int curr_gap_y = gap_y[row_offset + j];                                            \
                                                                                                   \
                int best = 0;                                                                      \
                best = diag_score > best ? diag_score : best;                                      \
                best = curr_gap_x > best ? curr_gap_x : best;                                      \
                best = curr_gap_y > best ? curr_gap_y : best;                                      \
                                                                                                   \
                match[row_offset + j] = best;                                                      \
                                                                                                   \
                int is_new_max = best > final_score;                                               \
                max_i = is_new_max * i + (1 - is_new_max) * max_i;                                 \
                max_j = is_new_max * j + (1 - is_new_max) * max_j;                                 \
                final_score = is_new_max * best + (1 - is_new_max) * final_score;                  \
            }                                                                                      \
        }                                                                                          \
    } while (0)

#ifdef USE_SIMD

INLINE void
simd_linear_row_init(int* matrix, int len1, int gap_penalty)
{
    veci_t indices = g_first_row_indices;
    veci_t gap_penalty_vec = set1_epi32(gap_penalty);

    for (int j = 1; j <= len1; j += NUM_ELEMS)
    {
        int remaining = len1 + 1 - j;
        if (remaining >= NUM_ELEMS)
        {
            veci_t values = mullo_epi32(indices, gap_penalty_vec);
            storeu((veci_t*)&matrix[j], values);
        }
        else
        {
            for (int k = 0; k < remaining; k++)
            {
                matrix[j + k] = (j + k) * gap_penalty;
            }
        }
        indices = add_epi32(indices, set1_epi32(NUM_ELEMS));
    }
}

INLINE void
simd_affine_global_row_init(int* match,
                            int* gap_x,
                            int* gap_y,
                            int len1,
                            int gap_start,
                            int gap_extend)
{
    veci_t indices = g_first_row_indices;
    veci_t gap_start_vec = set1_epi32(gap_start);
    veci_t gap_extend_vec = set1_epi32(gap_extend);
    veci_t int_min_half = set1_epi32(INT_MIN / 2);

    for (int j = 1; j <= len1; j += NUM_ELEMS)
    {
        int remaining = len1 + 1 - j;
        if (remaining >= NUM_ELEMS)
        {
            veci_t position_indices = add_epi32(indices, set1_epi32(j - 1));
            veci_t values = add_epi32(mullo_epi32(position_indices, gap_extend_vec), gap_start_vec);
            values = sub_epi32(setzero_si(), values);

            storeu((veci_t*)&match[j], values);
            storeu((veci_t*)&gap_x[j], values);
            storeu((veci_t*)&gap_y[j], int_min_half);
        }
        else
        {
            for (int k = 0; k < remaining; k++)
            {
                match[j + k] = -(gap_start + (j + k - 1) * gap_extend);
                gap_x[j + k] = match[j + k];
                gap_y[j + k] = INT_MIN / 2;
            }
        }
        indices = add_epi32(indices, set1_epi32(NUM_ELEMS));
    }
}

INLINE void
simd_affine_local_row_init(int* match, int* gap_x, int* gap_y, int len1)
{
    veci_t zero_vec = setzero_si();
    veci_t int_min_half = set1_epi32(INT_MIN / 2);

    for (int j = 0; j <= len1; j += NUM_ELEMS)
    {
        int remaining = len1 + 1 - j;
        if (remaining >= NUM_ELEMS)
        {
            storeu((veci_t*)&match[j], zero_vec);
            storeu((veci_t*)&gap_x[j], int_min_half);
            storeu((veci_t*)&gap_y[j], int_min_half);
        }
        else
        {
            for (int k = 0; k < remaining; k++)
            {
                match[j + k] = 0;
                gap_x[j + k] = gap_y[j + k] = INT_MIN / 2;
            }
        }
    }
}
#endif

INLINE int
align_nw(const char* seq1,
         const size_t len1,
         const char* seq2,
         const size_t len2,
         const ScoringMatrix* restrict scoring)
{
    size_t matrix_bytes = MATRIX_BYTES(len1, len2);
    int stack_matrix[USE_STACK_MATRIX(matrix_bytes) ? MATRIX_SIZE(len1, len2) : 1];
    int* restrict matrix = matrix_alloc(stack_matrix, matrix_bytes);
    const int cols = len1 + 1;
    const int gap_penalty = args_gap_penalty();

#ifdef USE_SIMD
    matrix[0] = 0;

    if (len1 >= NUM_ELEMS)
    {
        simd_linear_row_init(matrix, len1, gap_penalty);
    }
    else
    {
        for (int j = 1; j <= (int)len1; j++)
        {
            matrix[j] = j * gap_penalty;
        }
    }

    for (int i = 1; i <= (int)len2; i++)
    {
        matrix[i * cols] = i * gap_penalty;
    }
#else
    INIT_LINEAR_GLOBAL(matrix, cols, gap_penalty);
#endif

    SeqIndices seq1_indices = { 0 };
    seq_indices_precompute(&seq1_indices, seq1, len1);

    FILL_LINEAR_GLOBAL(matrix, cols, gap_penalty, similarity);

    int score = matrix[len2 * cols + len1];

    seq_indices_free(&seq1_indices);
    matrix_free(matrix, stack_matrix);

    return score;
}

INLINE int
align_ga(const char* seq1,
         const size_t len1,
         const char* seq2,
         const size_t len2,
         const ScoringMatrix* restrict scoring)
{
    size_t matrices_bytes = MATRICES_3X_BYTES(len1, len2);
    int stack_matrix[USE_STACK_MATRIX(matrices_bytes) ? 3 * MATRIX_SIZE(len1, len2) : 1];
    int* restrict matrix = matrix_alloc(stack_matrix, matrices_bytes);

    int* restrict match = matrix;
    int* restrict gap_x = matrix + MATRIX_SIZE(len1, len2);
    int* restrict gap_y = matrix + 2 * MATRIX_SIZE(len1, len2);
    const int cols = len1 + 1;
    const int gap_start = args_gap_start();
    const int gap_extend = args_gap_extend();

#ifdef USE_SIMD
    match[0] = 0;
    gap_x[0] = gap_y[0] = INT_MIN / 2;

    if (len1 >= NUM_ELEMS)
    {
        simd_affine_global_row_init(match, gap_x, gap_y, len1, gap_start, gap_extend);
    }
    else
    {
        for (int j = 1; j <= (int)len1; j++)
        {
            match[j] = -(gap_start + (j - 1) * gap_extend);
            gap_x[j] = match[j];
            gap_y[j] = INT_MIN / 2;
        }
    }

    for (int i = 1; i <= (int)len2; i++)
    {
        int idx = i * cols;
        match[idx] = -(gap_start + (i - 1) * gap_extend);
        gap_y[idx] = match[idx];
        gap_x[idx] = INT_MIN / 2;
    }
#else
    INIT_AFFINE_GLOBAL(match, gap_x, gap_y, cols, gap_start, gap_extend);
#endif

    SeqIndices seq1_indices = { 0 };
    seq_indices_precompute(&seq1_indices, seq1, len1);

    FILL_AFFINE_GLOBAL(match, gap_x, gap_y, cols, gap_start, gap_extend);

    int score = match[len2 * cols + len1];

    seq_indices_free(&seq1_indices);
    matrix_free(matrix, stack_matrix);

    return score;
}

INLINE int
align_sw(const char* seq1,
         const size_t len1,
         const char* seq2,
         const size_t len2,
         const ScoringMatrix* restrict scoring)
{
    size_t matrices_bytes = MATRICES_3X_BYTES(len1, len2);
    int stack_matrix[USE_STACK_MATRIX(matrices_bytes) ? 3 * MATRIX_SIZE(len1, len2) : 1];
    int* restrict matrix = matrix_alloc(stack_matrix, matrices_bytes);

    int* restrict match = matrix;
    int* restrict gap_x = matrix + MATRIX_SIZE(len1, len2);
    int* restrict gap_y = matrix + 2 * MATRIX_SIZE(len1, len2);
    const int cols = len1 + 1;
    const int gap_start = args_gap_start();
    const int gap_extend = args_gap_extend();

#ifdef USE_SIMD
    if (len1 >= NUM_ELEMS)
    {
        simd_affine_local_row_init(match, gap_x, gap_y, len1);
    }
    else
    {
        INIT_AFFINE_LOCAL(match, gap_x, gap_y, cols);
    }
#else
    INIT_AFFINE_LOCAL(match, gap_x, gap_y, cols);
#endif

    SeqIndices seq1_indices = { 0 };
    seq_indices_precompute(&seq1_indices, seq1, len1);

    int final_score = 0;
    int max_i = 0, max_j = 0;
    FILL_AFFINE_LOCAL(match, gap_x, gap_y, cols, gap_start, gap_extend, final_score, max_i, max_j);

    seq_indices_free(&seq1_indices);
    matrix_free(matrix, stack_matrix);

    return final_score;
}

INLINE int
align_pairwise(const char* seq1,
               const size_t len1,
               const char* seq2,
               const size_t len2,
               const ScoringMatrix* restrict scoring)
{
    switch (args_align_method())
    {
        case ALIGN_GOTOH_AFFINE:
            return align_ga(seq1, len1, seq2, len2, scoring);

        case ALIGN_SMITH_WATERMAN:
            return align_sw(seq1, len1, seq2, len2, scoring);

        case ALIGN_NEEDLEMAN_WUNSCH:
            return align_nw(seq1, len1, seq2, len2, scoring);

        default:
            UNREACHABLE();
    }
}

#endif // SEQALIGN_H