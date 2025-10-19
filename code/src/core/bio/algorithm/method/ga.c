#include "core/bio/algorithm/method/ga.h"

#include "core/bio/algorithm/global/affine.h"
#include "core/bio/algorithm/indices.h"
#include "core/bio/algorithm/matrix.h"
#include "util/print.h"

score_t
align_ga(const sequence_ptr_t seq1, const sequence_ptr_t seq2)
{
    const sequence_length_t len1 = seq1->length;
    const sequence_length_t len2 = seq2->length;

    size_t matrices_bytes = MATRICES_3X_BYTES(len1, len2);
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 6255)
    score_t* stack_matrix = USE_STACK_MATRIX(matrices_bytes) ? (score_t*)_alloca(matrices_bytes)
                                                             : NULL;
#pragma warning(pop)
#else
    score_t stack_matrix[USE_STACK_MATRIX(matrices_bytes) ? 3 * MATRIX_SIZE(len1, len2) : 1];
#endif
    score_t* restrict matrix = matrix_alloc(stack_matrix, matrices_bytes);

    score_t* restrict match = matrix;
    score_t* restrict gap_x = matrix + MATRIX_SIZE(len1, len2);
    score_t* restrict gap_y = matrix + 2 * MATRIX_SIZE(len1, len2);

    affine_global_init(match, gap_x, gap_y, seq1, seq2);

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

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 6255)
#endif
    seq1_indices.data = ALLOCA(seq1_indices.data, len1);
#ifdef _MSC_VER
#pragma warning(pop)
#endif
    seq1_indices.is_stack = true;

no_stack:
    seq_indices_precompute(&seq1_indices, seq1);

    affine_global_fill(match, gap_x, gap_y, &seq1_indices, seq1, seq2);

    score_t score = match[len2 * (len1 + 1U) + len1];

    seq_indices_free(&seq1_indices);
    matrix_free(matrix, stack_matrix);

    return score;
}
