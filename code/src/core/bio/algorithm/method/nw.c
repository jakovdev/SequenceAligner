#include "core/bio/algorithm/method/nw.h"

#include "core/app/args.h"
#include "core/bio/algorithm/global/linear.h"
#include "core/bio/algorithm/indices.h"
#include "core/bio/algorithm/matrix.h"
#include "util/print.h"

score_t
align_nw(sequence_ptr_t seq1, sequence_ptr_t seq2)
{
    const sequence_length_t len1 = seq1->length;
    const sequence_length_t len2 = seq2->length;

    size_t matrix_bytes = MATRIX_BYTES(len1, len2);
    size_t matrix_size = USE_STACK_MATRIX(matrix_bytes) ? MATRIX_SIZE(len1, len2) : 1;
#ifdef _MSC_VER
    score_t* stack_matrix = ALLOCA(stack_matrix, matrix_size);
#else
    score_t stack_matrix[matrix_size];
#endif
    score_t* restrict matrix = matrix_alloc(stack_matrix, matrix_bytes);

#ifdef USE_SIMD
    matrix[0] = 0;

    const sequence_length_t cols = len1 + 1U;
    const int gap_penalty = args_gap_penalty();

    if (len1 >= NUM_ELEMS)
    {
        simd_linear_global_row_init(matrix, seq1);
    }

    // TODO: Move below elsewhere
    else
    {
        for (sequence_length_t j = 1; j <= len1; j++)
        {
            matrix[j] = (score_t)j * (-gap_penalty);
        }
    }

    for (sequence_length_t i = 1; i <= len2; i++)
    {
        matrix[i * cols] = (score_t)i * (-gap_penalty);
    }

#else
    linear_global_init(matrix, seq1, seq2);
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

    linear_global_fill(matrix, &seq1_indices, seq1, seq2);

    score_t score = matrix[len2 * (len1 + 1U) + len1];

    seq_indices_free(&seq1_indices);
    matrix_free(matrix, stack_matrix);

    return score;
}
