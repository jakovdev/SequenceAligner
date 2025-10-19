#include "core/bio/algorithm/local/affine.h"

#include "core/app/args.h"
#include "core/bio/algorithm/indices.h"
#include "core/bio/score/scoring.h"
#include "system/arch.h"

void
affine_local_init(score_t* restrict match,
                  score_t* restrict gap_x,
                  score_t* restrict gap_y,
                  const sequence_ptr_t seq1,
                  const sequence_ptr_t seq2)
{
    const sequence_length_t len1 = seq1->length;
    const sequence_length_t len2 = seq2->length;
    const sequence_length_t cols = len1 + 1U;

    match[0] = 0;
    gap_x[0] = gap_y[0] = SCORE_MIN;

    UNROLL(8) for (sequence_length_t j = 1; j <= len1; j++)
    {
        match[j] = 0;
        gap_x[j] = gap_y[j] = SCORE_MIN;
    }

    UNROLL(8) for (sequence_length_t i = 1; i <= len2; i++)
    {
        sequence_length_t idx = i * cols;
        match[idx] = 0;
        gap_x[idx] = gap_y[idx] = SCORE_MIN;
    }
}

score_t
affine_local_fill(score_t* restrict match,
                  score_t* restrict gap_x,
                  score_t* restrict gap_y,
                  const SeqIndices* seq1_indices,
                  const sequence_ptr_t seq1,
                  const sequence_ptr_t seq2)
{
    const sequence_length_t len1 = seq1->length;
    const sequence_length_t len2 = seq2->length;
    const sequence_length_t cols = len1 + 1U;

    const int gap_open = args_gap_open();
    const int gap_extend = args_gap_extend();

    const int* restrict seq1_idx = seq1_indices->data;
    score_t score = 0;

    for (sequence_length_t i = 1; i <= len2; ++i)
    {
        sequence_length_t row_offset = i * cols;
        sequence_length_t prev_row_offset = (i - 1) * cols;
        int c2_idx = SEQUENCE_LOOKUP[(unsigned char)seq2->letters[i - 1]];

        prefetch(&match[row_offset + PREFETCH_DISTANCE]);
        prefetch(&gap_x[row_offset + PREFETCH_DISTANCE]);
        prefetch(&gap_y[row_offset + PREFETCH_DISTANCE]);

        for (sequence_length_t j = 1; j <= len1; j++)
        {
            score_t similarity = SCORING_MATRIX[seq1_idx[j - 1]][c2_idx];
            score_t diag_score = match[prev_row_offset + j - 1] + similarity;

            score_t prev_match_x = match[row_offset + j - 1];
            score_t prev_gap_x = gap_x[row_offset + j - 1];
            score_t prev_match_y = match[prev_row_offset + j];
            score_t prev_gap_y = gap_y[prev_row_offset + j];

            score_t open_x = prev_match_x - (gap_open);
            score_t extend_x = prev_gap_x - (gap_extend);
            score_t open_y = prev_match_y - (gap_open);
            score_t extend_y = prev_gap_y - (gap_extend);

            gap_x[row_offset + j] = (open_x > extend_x) ? open_x : extend_x;
            gap_y[row_offset + j] = (open_y > extend_y) ? open_y : extend_y;

            score_t curr_gap_x = gap_x[row_offset + j];
            score_t curr_gap_y = gap_y[row_offset + j];

            score_t best = 0;
            best = diag_score > best ? diag_score : best;
            best = curr_gap_x > best ? curr_gap_x : best;
            best = curr_gap_y > best ? curr_gap_y : best;

            match[row_offset + j] = best;

            if (best > score)
            {
                score = best;
            }
        }
    }

    return score;
}

#ifdef USE_SIMD

void
simd_affine_local_row_init(score_t* restrict match,
                           score_t* restrict gap_x,
                           score_t* restrict gap_y,
                           const sequence_ptr_t seq1,
                           const sequence_ptr_t seq2)
{
    const sequence_length_t len1 = seq1->length;
    const sequence_length_t len2 = seq2->length;

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
