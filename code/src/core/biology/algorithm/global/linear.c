#include "core/biology/algorithm/global/linear.h"

#include "core/app/args.h"
#include "core/biology/algorithm/indices.h"
#include "core/biology/score/scoring.h"
#include "system/arch.h"

void
linear_global_init(score_t* restrict matrix, const sequence_ptr_t seq1, const sequence_ptr_t seq2)
{
    const sequence_length_t len1 = seq1->length;
    const sequence_length_t len2 = seq2->length;
    const sequence_length_t cols = len1 + 1U;

    const int gap_penalty = args_gap_penalty();

    matrix[0] = 0;
    VECTORIZE UNROLL(8) for (sequence_length_t j = 1; j <= len1; j++)
    {
        matrix[j] = j * (-gap_penalty);
    }

    VECTORIZE UNROLL(8) for (sequence_length_t i = 1; i <= len2; i++)
    {
        matrix[i * cols] = i * (-gap_penalty);
    }
}

void
linear_global_fill(score_t* restrict matrix,
                   const SeqIndices* seq1_indices,
                   const sequence_ptr_t seq1,
                   const sequence_ptr_t seq2)
{
    const sequence_length_t len1 = seq1->length;
    const sequence_length_t len2 = seq2->length;
    const sequence_length_t cols = len1 + 1U;

    const int gap_penalty = args_gap_penalty();

    for (sequence_length_t i = 1; i <= len2; ++i)
    {
        sequence_length_t row_offset = i * cols;
        sequence_length_t prev_row_offset = (i - 1) * cols;
        int c2_idx = SEQUENCE_LOOKUP[(unsigned char)seq2->letters[i - 1]];

        prefetch(&matrix[row_offset + PREFETCH_DISTANCE]);

        UNROLL(4) for (sequence_length_t j = 1; j <= len1; j++)
        {
            score_t match = matrix[prev_row_offset + j - 1] +
                            SCORING_MATRIX[seq1_indices->data[j - 1]][c2_idx];
            score_t del = matrix[prev_row_offset + j] + (-gap_penalty);
            score_t insert = matrix[row_offset + j - 1] + (-gap_penalty);
            matrix[row_offset + j] = match > del ? (match > insert ? match : insert)
                                                 : (del > insert ? del : insert);
        }
    }
}

#ifdef USE_SIMD

void
simd_linear_global_row_init(score_t* restrict matrix, const sequence_ptr_t seq1)
{
    const sequence_length_t len1 = seq1->length;

    const int gap_penalty = args_gap_penalty();

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

#endif