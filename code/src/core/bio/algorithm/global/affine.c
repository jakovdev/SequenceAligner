#include "core/bio/algorithm/global/affine.h"

#include "core/app/args.h"
#include "core/bio/algorithm/indices.h"
#include "core/bio/score/scoring.h"
#include "core/bio/types.h"
#include "system/arch.h"
#include "system/simd.h"

void affine_global_init(score_t *restrict match, score_t *restrict gap_x,
			score_t *restrict gap_y, sequence_ptr_t seq1,
			sequence_ptr_t seq2)
{
	match[0] = 0;
	gap_x[0] = gap_y[0] = SCORE_MIN;

	const sequence_length_t len1 = seq1->length;
	const sequence_length_t len2 = seq2->length;
	const sequence_length_t cols = len1 + 1U;
	const int gap_open = args_gap_open();
	const int gap_extend = args_gap_extend();

	UNROLL(8) for (sequence_length_t j = 1; j <= len1; j++)
	{
		gap_x[j] =
			MAX(match[j - 1] - gap_open, gap_x[j - 1] - gap_extend);
		match[j] = gap_x[j];
		gap_y[j] = SCORE_MIN;
	}

	UNROLL(8) for (sequence_length_t i = 1; i <= len2; i++)
	{
		sequence_length_t idx = i * cols;
		gap_y[idx] = MAX(match[idx - cols] - gap_open,
				 gap_y[idx - cols] - gap_extend);
		match[idx] = gap_y[idx];
		gap_x[idx] = SCORE_MIN;
	}
}

void affine_global_fill(score_t *restrict match, score_t *restrict gap_x,
			score_t *restrict gap_y, const SeqIndices *seq1_indices,
			sequence_ptr_t seq1, sequence_ptr_t seq2)
{
	const sequence_length_t len1 = seq1->length;
	const sequence_length_t len2 = seq2->length;
	const sequence_length_t cols = len1 + 1U;
	const int gap_open = args_gap_open();
	const int gap_extend = args_gap_extend();

	for (sequence_length_t i = 1; i <= len2; ++i) {
		sequence_length_t row_offset = i * cols;
		sequence_length_t prev_row_offset = (i - 1) * cols;
		int c2_idx =
			SEQUENCE_LOOKUP[(unsigned char)seq2->letters[i - 1]];

		prefetch(&match[row_offset + PREFETCH_DISTANCE]);
		prefetch(&gap_x[row_offset + PREFETCH_DISTANCE]);
		prefetch(&gap_y[row_offset + PREFETCH_DISTANCE]);

		const int *restrict seq1_idx_data = seq1_indices->data;
		VECTORIZE for (sequence_length_t j = 1; j <= len1; j++)
		{
			score_t similarity =
				SCORING_MATRIX[seq1_idx_data[j - 1]][c2_idx];
			score_t diag_score =
				match[prev_row_offset + j - 1] + similarity;

			score_t prev_match_x = match[row_offset + j - 1];
			score_t prev_gap_x = gap_x[row_offset + j - 1];
			score_t prev_match_y = match[prev_row_offset + j];
			score_t prev_gap_y = gap_y[prev_row_offset + j];

			score_t open_x = prev_match_x - gap_open;
			score_t extend_x = prev_gap_x - gap_extend;
			gap_x[row_offset + j] = (open_x > extend_x) ? open_x :
								      extend_x;

			score_t open_y = prev_match_y - gap_open;
			score_t extend_y = prev_gap_y - gap_extend;
			gap_y[row_offset + j] = (open_y > extend_y) ? open_y :
								      extend_y;

			score_t best = diag_score;
			if (gap_x[row_offset + j] > best)
				best = gap_x[row_offset + j];
			if (gap_y[row_offset + j] > best)
				best = gap_y[row_offset + j];
			match[row_offset + j] = best;
		}
	}
}
