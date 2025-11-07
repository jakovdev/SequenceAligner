#include "core/bio/algorithm/local/affine.h"

#include "core/app/args.h"
#include "core/bio/score/scoring.h"
#include "system/compiler.h"
#include "system/simd.h"

void affine_local_init(s32 *restrict match, s32 *restrict gap_x,
		       s32 *restrict gap_y, sequence_ptr_t seq1,
		       sequence_ptr_t seq2)
{
	const u64 len1 = seq1->length;
	const u64 len2 = seq2->length;
	const u64 cols = len1 + 1;

	match[0] = 0;
	gap_x[0] = gap_y[0] = SCORE_MIN;
	UNROLL(8) for (u64 j = 1; j <= len1; j++)
	{
		match[j] = 0;
		gap_x[j] = gap_y[j] = SCORE_MIN;
	}

	UNROLL(8) for (u64 i = 1; i <= len2; i++)
	{
		u64 idx = i * cols;
		match[idx] = 0;
		gap_x[idx] = gap_y[idx] = SCORE_MIN;
	}
}

s32 affine_local_fill(s32 *restrict match, s32 *restrict gap_x,
		      s32 *restrict gap_y, const s32 *restrict seq1_indices,
		      sequence_ptr_t seq1, sequence_ptr_t seq2)
{
	const u64 len1 = seq1->length;
	const u64 len2 = seq2->length;
	const u64 cols = len1 + 1;
	const s32 gap_open = args_gap_open();
	const s32 gap_extend = args_gap_extend();
	s32 score = 0;
	for (u64 i = 1; i <= len2; ++i) {
		const u64 row_offset = i * cols;
		const u64 prev_row_offset = (i - 1) * cols;
		const int c2_idx = SEQUENCE_LOOKUP[(uchar)seq2->letters[i - 1]];

		prefetch(&match[row_offset + PREFETCH_DISTANCE]);
		prefetch(&gap_x[row_offset + PREFETCH_DISTANCE]);
		prefetch(&gap_y[row_offset + PREFETCH_DISTANCE]);

		for (u64 j = 1; j <= len1; j++) {
			const s32 similarity =
				SCORING_MATRIX[seq1_indices[j - 1]][c2_idx];
			const s32 diag_score =
				match[prev_row_offset + j - 1] + similarity;

			const s32 prev_match_x = match[row_offset + j - 1];
			const s32 prev_gap_x = gap_x[row_offset + j - 1];
			const s32 prev_match_y = match[prev_row_offset + j];
			const s32 prev_gap_y = gap_y[prev_row_offset + j];

			const s32 open_x = prev_match_x - (gap_open);
			const s32 extend_x = prev_gap_x - (gap_extend);
			const s32 open_y = prev_match_y - (gap_open);
			const s32 extend_y = prev_gap_y - (gap_extend);

			gap_x[row_offset + j] = (open_x > extend_x) ? open_x :
								      extend_x;
			gap_y[row_offset + j] = (open_y > extend_y) ? open_y :
								      extend_y;

			const s32 curr_gap_x = gap_x[row_offset + j];
			const s32 curr_gap_y = gap_y[row_offset + j];

			s32 best = 0;
			best = diag_score > best ? diag_score : best;
			best = curr_gap_x > best ? curr_gap_x : best;
			best = curr_gap_y > best ? curr_gap_y : best;
			match[row_offset + j] = best;
			if (best > score)
				score = best;
		}
	}

	return score;
}

#if USE_SIMD == 1

void simd_affine_local_row_init(s32 *restrict match, s32 *restrict gap_x,
				s32 *restrict gap_y, sequence_ptr_t seq1,
				sequence_ptr_t seq2)
{
	const u64 len1 = seq1->length;
	const u64 len2 = seq2->length;

	veci_t zero_vec = setzero_si();
	veci_t score_min = set1_epi32(SCORE_MIN);

	VECTORIZE for (u64 j = 0; j <= len1; j += NUM_ELEMS)
	{
		u64 remaining = len1 + 1 - j;
		if (remaining >= NUM_ELEMS) {
			storeu((veci_t *)&match[j], zero_vec);
			storeu((veci_t *)&gap_x[j], score_min);
			storeu((veci_t *)&gap_y[j], score_min);
		} else {
			for (u64 k = 0; k < remaining; k++) {
				match[j + k] = 0;
				gap_x[j + k] = gap_y[j + k] = SCORE_MIN;
			}
		}
	}

	VECTORIZE for (u64 i = 1; i <= len2; i++)
	{
		u64 idx = i * (len1 + 1);
		match[idx] = 0;
		gap_x[idx] = SCORE_MIN;
		gap_y[idx] = SCORE_MIN;
	}
}

#endif
