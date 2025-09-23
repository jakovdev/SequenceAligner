#pragma once
#ifndef SEQALIGN_H
#define SEQALIGN_H

#include "args.h"
#include "biotypes.h"
#include "print.h"
#include "scoring.h"

#define MAX_STACK_SEQUENCE_LENGTH (4 * KiB)
#define STACK_MATRIX_THRESHOLD (128 * KiB)
#define MATRIX_SIZE(len1, len2) ((len1 + 1) * (len2 + 1))
#define MATRIX_BYTES(len1, len2) (MATRIX_SIZE(len1, len2) * sizeof(score_t))
#define MATRICES_3X_BYTES(len1, len2) (3 * MATRIX_BYTES(len1, len2))
#define USE_STACK_MATRIX(bytes) ((bytes) <= STACK_MATRIX_THRESHOLD)

typedef struct
{
    int* data;
    bool is_stack;
} SeqIndices;

static inline void
seq_indices_precompute(SeqIndices* indices, const sequence_t seq)
{
    char* restrict seq_letters = seq.letters;
    sequence_length_t len = seq.length;
    int* restrict data = indices->data;
#ifdef USE_SIMD
    const sequence_length_t vector_len = (len / BYTES) * BYTES;
    sequence_length_t i = 0;

    for (; i < vector_len; i += BYTES)
    {
        prefetch(seq_letters + i + BYTES * 2);

        VECTORIZE for (sequence_length_t j = 0; j < BYTES; j++)
        {
            data[i + j] = SEQUENCE_LOOKUP[(unsigned char)seq_letters[i + j]];
        }
    }

    for (; i < len; i++)
    {
        data[i] = SEQUENCE_LOOKUP[(unsigned char)seq_letters[i]];
    }

#else

    VECTORIZE UNROLL(8) for (sequence_length_t i = 0; i < len; ++i)
    {
        data[i] = SEQUENCE_LOOKUP[(unsigned char)seq_letters[i]];
    }

#endif
}

static inline void
seq_indices_free(SeqIndices* indices)
{
    if (!indices->is_stack && indices->data)
    {
        free(indices->data);
        indices->data = NULL;
    }
}

static inline score_t*
matrix_alloc(score_t* stack_matrix, size_t bytes)
{
    if (USE_STACK_MATRIX(bytes))
    {
        return stack_matrix;
    }

    else
    {
        return (score_t*)(alloc_huge_page(bytes));
    }
}

static inline void
matrix_free(score_t* matrix, score_t* stack_matrix)
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
        VECTORIZE UNROLL(8) for (sequence_length_t j = 1; j <= len1; j++)                          \
        {                                                                                          \
            matrix[j] = j * (-gap_penalty);                                                        \
        }                                                                                          \
                                                                                                   \
        VECTORIZE UNROLL(8) for (sequence_length_t i = 1; i <= len2; i++)                          \
        {                                                                                          \
            matrix[i * cols] = i * (-gap_penalty);                                                 \
        }                                                                                          \
    } while (false)

#define INIT_AFFINE_GLOBAL(match, gap_x, gap_y, cols, gap_open, gap_extend)                        \
    do                                                                                             \
    {                                                                                              \
        match[0] = 0;                                                                              \
        gap_x[0] = gap_y[0] = SCORE_MIN;                                                           \
                                                                                                   \
        UNROLL(8) for (sequence_length_t j = 1; j <= len1; j++)                                    \
        {                                                                                          \
            gap_x[j] = MAX(match[j - 1] - gap_open, gap_x[j - 1] - gap_extend);                    \
            match[j] = gap_x[j];                                                                   \
            gap_y[j] = SCORE_MIN;                                                                  \
        }                                                                                          \
                                                                                                   \
        UNROLL(8) for (sequence_length_t i = 1; i <= len2; i++)                                    \
        {                                                                                          \
            sequence_length_t idx = i * cols;                                                      \
            gap_y[idx] = MAX(match[idx - cols] - gap_open, gap_y[idx - cols] - gap_extend);        \
            match[idx] = gap_y[idx];                                                               \
            gap_x[idx] = SCORE_MIN;                                                                \
        }                                                                                          \
    } while (false)

#define INIT_AFFINE_LOCAL(match, gap_x, gap_y, cols)                                               \
    do                                                                                             \
    {                                                                                              \
        match[0] = 0;                                                                              \
        gap_x[0] = gap_y[0] = SCORE_MIN;                                                           \
                                                                                                   \
        UNROLL(8) for (sequence_length_t j = 1; j <= len1; j++)                                    \
        {                                                                                          \
            match[j] = 0;                                                                          \
            gap_x[j] = gap_y[j] = SCORE_MIN;                                                       \
        }                                                                                          \
                                                                                                   \
        UNROLL(8) for (sequence_length_t i = 1; i <= len2; i++)                                    \
        {                                                                                          \
            sequence_length_t idx = i * cols;                                                      \
            match[idx] = 0;                                                                        \
            gap_x[idx] = gap_y[idx] = SCORE_MIN;                                                   \
        }                                                                                          \
    } while (false)

#define FILL_LINEAR_GLOBAL(matrix, cols, gap_penalty)                                              \
    do                                                                                             \
    {                                                                                              \
        for (sequence_length_t i = 1; i <= len2; ++i)                                              \
        {                                                                                          \
            sequence_length_t row_offset = i * cols;                                               \
            sequence_length_t prev_row_offset = (i - 1) * cols;                                    \
            int c2_idx = SEQUENCE_LOOKUP[(unsigned char)seq2_letters[i - 1]];                      \
                                                                                                   \
            prefetch(&matrix[row_offset + PREFETCH_DISTANCE]);                                     \
                                                                                                   \
            UNROLL(4) for (sequence_length_t j = 1; j <= len1; j++)                                \
            {                                                                                      \
                score_t match = matrix[prev_row_offset + j - 1] +                                  \
                                SCORING_MATRIX[seq1_indices.data[j - 1]][c2_idx];                  \
                score_t del = matrix[prev_row_offset + j] + (-gap_penalty);                        \
                score_t insert = matrix[row_offset + j - 1] + (-gap_penalty);                      \
                matrix[row_offset + j] = match > del ? (match > insert ? match : insert)           \
                                                     : (del > insert ? del : insert);              \
            }                                                                                      \
        }                                                                                          \
    } while (false)

#define FILL_AFFINE_GLOBAL(match, gap_x, gap_y, cols, gap_open, gap_extend)                        \
    do                                                                                             \
    {                                                                                              \
        for (sequence_length_t i = 1; i <= len2; ++i)                                              \
        {                                                                                          \
            sequence_length_t row_offset = i * cols;                                               \
            sequence_length_t prev_row_offset = (i - 1) * cols;                                    \
            int c2_idx = SEQUENCE_LOOKUP[(unsigned char)seq2_letters[i - 1]];                      \
                                                                                                   \
            prefetch(&match[row_offset + PREFETCH_DISTANCE]);                                      \
            prefetch(&gap_x[row_offset + PREFETCH_DISTANCE]);                                      \
            prefetch(&gap_y[row_offset + PREFETCH_DISTANCE]);                                      \
                                                                                                   \
            const int* restrict seq1_idx_data = seq1_indices.data;                                 \
                                                                                                   \
            VECTORIZE for (sequence_length_t j = 1; j <= len1; j++)                                \
            {                                                                                      \
                score_t similarity = SCORING_MATRIX[seq1_idx_data[j - 1]][c2_idx];                 \
                score_t diag_score = match[prev_row_offset + j - 1] + similarity;                  \
                                                                                                   \
                score_t prev_match_x = match[row_offset + j - 1];                                  \
                score_t prev_gap_x = gap_x[row_offset + j - 1];                                    \
                score_t prev_match_y = match[prev_row_offset + j];                                 \
                score_t prev_gap_y = gap_y[prev_row_offset + j];                                   \
                                                                                                   \
                score_t open_x = prev_match_x - gap_open;                                          \
                score_t extend_x = prev_gap_x - gap_extend;                                        \
                gap_x[row_offset + j] = (open_x > extend_x) ? open_x : extend_x;                   \
                                                                                                   \
                score_t open_y = prev_match_y - gap_open;                                          \
                score_t extend_y = prev_gap_y - gap_extend;                                        \
                gap_y[row_offset + j] = (open_y > extend_y) ? open_y : extend_y;                   \
                                                                                                   \
                score_t best = diag_score;                                                         \
                if (gap_x[row_offset + j] > best)                                                  \
                    best = gap_x[row_offset + j];                                                  \
                if (gap_y[row_offset + j] > best)                                                  \
                    best = gap_y[row_offset + j];                                                  \
                                                                                                   \
                match[row_offset + j] = best;                                                      \
            }                                                                                      \
        }                                                                                          \
    } while (false)

#define FILL_AFFINE_LOCAL(match, gap_x, gap_y, cols, gap_open, gap_extend, score)                  \
    do                                                                                             \
    {                                                                                              \
        const int* restrict seq1_idx = seq1_indices.data;                                          \
        score = 0;                                                                                 \
                                                                                                   \
        for (sequence_length_t i = 1; i <= len2; ++i)                                              \
        {                                                                                          \
            sequence_length_t row_offset = i * cols;                                               \
            sequence_length_t prev_row_offset = (i - 1) * cols;                                    \
            int c2_idx = SEQUENCE_LOOKUP[(unsigned char)seq2_letters[i - 1]];                      \
                                                                                                   \
            prefetch(&match[row_offset + PREFETCH_DISTANCE]);                                      \
            prefetch(&gap_x[row_offset + PREFETCH_DISTANCE]);                                      \
            prefetch(&gap_y[row_offset + PREFETCH_DISTANCE]);                                      \
                                                                                                   \
            for (sequence_length_t j = 1; j <= len1; j++)                                          \
            {                                                                                      \
                score_t similarity = SCORING_MATRIX[seq1_idx[j - 1]][c2_idx];                      \
                score_t diag_score = match[prev_row_offset + j - 1] + similarity;                  \
                                                                                                   \
                score_t prev_match_x = match[row_offset + j - 1];                                  \
                score_t prev_gap_x = gap_x[row_offset + j - 1];                                    \
                score_t prev_match_y = match[prev_row_offset + j];                                 \
                score_t prev_gap_y = gap_y[prev_row_offset + j];                                   \
                                                                                                   \
                score_t open_x = prev_match_x - (gap_open);                                        \
                score_t extend_x = prev_gap_x - (gap_extend);                                      \
                score_t open_y = prev_match_y - (gap_open);                                        \
                score_t extend_y = prev_gap_y - (gap_extend);                                      \
                                                                                                   \
                gap_x[row_offset + j] = (open_x > extend_x) ? open_x : extend_x;                   \
                gap_y[row_offset + j] = (open_y > extend_y) ? open_y : extend_y;                   \
                                                                                                   \
                score_t curr_gap_x = gap_x[row_offset + j];                                        \
                score_t curr_gap_y = gap_y[row_offset + j];                                        \
                                                                                                   \
                score_t best = 0;                                                                  \
                best = diag_score > best ? diag_score : best;                                      \
                best = curr_gap_x > best ? curr_gap_x : best;                                      \
                best = curr_gap_y > best ? curr_gap_y : best;                                      \
                                                                                                   \
                match[row_offset + j] = best;                                                      \
                                                                                                   \
                if (best > score)                                                                  \
                {                                                                                  \
                    score = best;                                                                  \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    } while (false)

#ifdef USE_SIMD

static inline void
simd_linear_row_init(score_t* restrict matrix, sequence_length_t len1, int gap_penalty)
{
    veci_t indices = g_first_row_indices;
    veci_t gap_penalty_vec = set1_epi32(-gap_penalty);

    for (sequence_length_t j = 1; j <= len1; j += NUM_ELEMS)
    {
        sequence_length_t remaining = len1 + 1U - j;
        if (remaining >= NUM_ELEMS)
        {
            veci_t values = mullo_epi32(indices, gap_penalty_vec);
            storeu((veci_t*)&matrix[j], values);
        }

        else
        {
            for (sequence_length_t k = 0; k < remaining; k++)
            {
                matrix[j + k] = (j + k) * (-gap_penalty);
            }
        }

        indices = add_epi32(indices, set1_epi32(NUM_ELEMS));
    }
}

static inline void
simd_affine_local_row_init(score_t* restrict match,
                           score_t* restrict gap_x,
                           score_t* restrict gap_y,
                           sequence_length_t len1,
                           sequence_length_t len2)
{
    veci_t zero_vec = setzero_si();
    veci_t score_min = set1_epi32(SCORE_MIN);

    VECTORIZE for (sequence_length_t j = 0; j <= len1; j += NUM_ELEMS)
    {
        sequence_length_t remaining = len1 + 1U - j;
        if (remaining >= NUM_ELEMS)
        {
            storeu((veci_t*)&match[j], zero_vec);
            storeu((veci_t*)&gap_x[j], score_min);
            storeu((veci_t*)&gap_y[j], score_min);
        }

        else
        {
            for (sequence_length_t k = 0; k < remaining; k++)
            {
                match[j + k] = 0;
                gap_x[j + k] = gap_y[j + k] = SCORE_MIN;
            }
        }
    }

    VECTORIZE for (sequence_length_t i = 1; i <= len2; i++)
    {
        sequence_length_t idx = i * (len1 + 1U);
        match[idx] = 0;
        gap_x[idx] = SCORE_MIN;
        gap_y[idx] = SCORE_MIN;
    }
}

#endif

static inline score_t
align_nw(const sequence_t seq1, const sequence_t seq2)
{
    const char* restrict seq2_letters = seq2.letters;
    const sequence_length_t len1 = seq1.length;
    const sequence_length_t len2 = seq2.length;

    size_t matrix_bytes = MATRIX_BYTES(len1, len2);
    score_t stack_matrix[USE_STACK_MATRIX(matrix_bytes) ? MATRIX_SIZE(len1, len2) : 1];
    score_t* restrict matrix = matrix_alloc(stack_matrix, matrix_bytes);
    const sequence_length_t cols = len1 + 1;
    const int gap_penalty = args_gap_penalty();

#ifdef USE_SIMD
    matrix[0] = 0;

    if (len1 >= NUM_ELEMS)
    {
        simd_linear_row_init(matrix, len1, gap_penalty);
    }

    else
    {
        for (sequence_length_t j = 1; j <= len1; j++)
        {
            matrix[j] = j * (-gap_penalty);
        }
    }

    for (sequence_length_t i = 1; i <= len2; i++)
    {
        matrix[i * cols] = i * (-gap_penalty);
    }

#else
    INIT_LINEAR_GLOBAL(matrix, cols, gap_penalty);
#endif

    SeqIndices seq1_indices = { 0 };

    if (len1 > MAX_STACK_SEQUENCE_LENGTH)
    {
        seq1_indices.data = MALLOC(seq1_indices.data, len1);
        if (!seq1_indices.data)
        {
            print_error_prefix("SEQALIGN - NW");
            print(ERROR, MSG_NONE, "Failed to allocate memory for sequence indices");
            exit(1);
        }

        seq1_indices.is_stack = false;
        goto no_stack;
    }

    seq1_indices.data = ALLOCA(seq1_indices.data, len1);
    seq1_indices.is_stack = true;

no_stack:
    seq_indices_precompute(&seq1_indices, seq1);

    FILL_LINEAR_GLOBAL(matrix, cols, gap_penalty);

    score_t score = matrix[len2 * cols + len1];

    seq_indices_free(&seq1_indices);
    matrix_free(matrix, stack_matrix);

    return score;
}

static inline score_t
align_ga(const sequence_t seq1, const sequence_t seq2)
{
    const char* restrict seq2_letters = seq2.letters;
    const sequence_length_t len1 = seq1.length;
    const sequence_length_t len2 = seq2.length;

    size_t matrices_bytes = MATRICES_3X_BYTES(len1, len2);
    score_t stack_matrix[USE_STACK_MATRIX(matrices_bytes) ? 3 * MATRIX_SIZE(len1, len2) : 1];
    score_t* restrict matrix = matrix_alloc(stack_matrix, matrices_bytes);

    score_t* restrict match = matrix;
    score_t* restrict gap_x = matrix + MATRIX_SIZE(len1, len2);
    score_t* restrict gap_y = matrix + 2 * MATRIX_SIZE(len1, len2);
    const sequence_length_t cols = len1 + 1;
    const int gap_open = args_gap_open();
    const int gap_extend = args_gap_extend();

    INIT_AFFINE_GLOBAL(match, gap_x, gap_y, cols, gap_open, gap_extend);

    SeqIndices seq1_indices = { 0 };

    if (len1 > MAX_STACK_SEQUENCE_LENGTH)
    {
        seq1_indices.data = MALLOC(seq1_indices.data, len1);
        if (!seq1_indices.data)
        {
            print_error_prefix("SEQALIGN - GA");
            print(ERROR, MSG_NONE, "Failed to allocate memory for sequence indices");
            exit(1);
        }

        seq1_indices.is_stack = false;
        goto no_stack;
    }

    seq1_indices.data = ALLOCA(seq1_indices.data, len1);
    seq1_indices.is_stack = true;

no_stack:
    seq_indices_precompute(&seq1_indices, seq1);

    FILL_AFFINE_GLOBAL(match, gap_x, gap_y, cols, gap_open, gap_extend);

    score_t score = match[len2 * cols + len1];

    seq_indices_free(&seq1_indices);
    matrix_free(matrix, stack_matrix);

    return score;
}

static inline score_t
align_sw(const sequence_t seq1, const sequence_t seq2)
{
    const char* restrict seq2_letters = seq2.letters;
    const sequence_length_t len1 = seq1.length;
    const sequence_length_t len2 = seq2.length;

    size_t matrices_bytes = MATRICES_3X_BYTES(len1, len2);
    score_t stack_matrix[USE_STACK_MATRIX(matrices_bytes) ? 3 * MATRIX_SIZE(len1, len2) : 1];
    score_t* restrict matrix = matrix_alloc(stack_matrix, matrices_bytes);

    score_t* restrict match = matrix;
    score_t* restrict gap_x = matrix + MATRIX_SIZE(len1, len2);
    score_t* restrict gap_y = matrix + 2 * MATRIX_SIZE(len1, len2);
    const sequence_length_t cols = len1 + 1;
    const int gap_open = args_gap_open();
    const int gap_extend = args_gap_extend();

#ifdef USE_SIMD
    if (len1 >= NUM_ELEMS)
    {
        simd_affine_local_row_init(match, gap_x, gap_y, len1, len2);
    }

    else
    {
        INIT_AFFINE_LOCAL(match, gap_x, gap_y, cols);
    }

#else
    INIT_AFFINE_LOCAL(match, gap_x, gap_y, cols);
#endif

    SeqIndices seq1_indices = { 0 };

    if (len1 > MAX_STACK_SEQUENCE_LENGTH)
    {
        seq1_indices.data = MALLOC(seq1_indices.data, len1);
        if (!seq1_indices.data)
        {
            print_error_prefix("SEQALIGN - SW");
            print(ERROR, MSG_NONE, "Failed to allocate memory for sequence indices");
            exit(1);
        }

        seq1_indices.is_stack = false;
        goto no_stack;
    }

    seq1_indices.data = ALLOCA(seq1_indices.data, len1);
    seq1_indices.is_stack = true;

no_stack:
    seq_indices_precompute(&seq1_indices, seq1);

    score_t score = 0;
    FILL_AFFINE_LOCAL(match, gap_x, gap_y, cols, gap_open, gap_extend, score);

    seq_indices_free(&seq1_indices);
    matrix_free(matrix, stack_matrix);

    return score;
}

static inline score_t
align_pairwise(const sequence_t seq1, const sequence_t seq2)
{
    switch (args_align_method())
    {
        case ALIGN_GOTOH_AFFINE:
            return align_ga(seq1, seq2);

        case ALIGN_SMITH_WATERMAN:
            return align_sw(seq1, seq2);

        case ALIGN_NEEDLEMAN_WUNSCH:
            return align_nw(seq1, seq2);

        default:
            UNREACHABLE();
    }
}

#endif // SEQALIGN_H