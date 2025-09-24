#include "core/bio/algorithm/method/sw.h"

#include "core/bio/algorithm/indices.h"
#include "core/bio/algorithm/local/affine.h"
#include "core/bio/algorithm/matrix.h"
#include "util/print.h"

score_t
align_sw(const sequence_ptr_t seq1, const sequence_ptr_t seq2)
{
    const sequence_length_t len1 = seq1->length;
    const sequence_length_t len2 = seq2->length;

    size_t matrices_bytes = MATRICES_3X_BYTES(len1, len2);
    score_t stack_matrix[USE_STACK_MATRIX(matrices_bytes) ? 3 * MATRIX_SIZE(len1, len2) : 1];
    score_t* restrict matrix = matrix_alloc(stack_matrix, matrices_bytes);

    score_t* restrict match = matrix;
    score_t* restrict gap_x = matrix + MATRIX_SIZE(len1, len2);
    score_t* restrict gap_y = matrix + 2 * MATRIX_SIZE(len1, len2);

#ifdef USE_SIMD
    if (len1 >= NUM_ELEMS)
    {
        simd_affine_local_row_init(match, gap_x, gap_y, seq1, seq2);
    }

    else
#endif
    {
        affine_local_init(match, gap_x, gap_y, seq1, seq2);
    }

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

    score_t score = affine_local_fill(match, gap_x, gap_y, &seq1_indices, seq1, seq2);

    seq_indices_free(&seq1_indices);
    matrix_free(matrix, stack_matrix);

    return score;
}