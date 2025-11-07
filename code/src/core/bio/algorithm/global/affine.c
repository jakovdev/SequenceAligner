#include "core/bio/algorithm/global/affine.h"

#include "core/app/args.h"
#include "core/bio/score/scoring.h"
#include "core/bio/types.h"
#include "system/compiler.h"
#include "system/os.h"
#include "system/simd.h"

void affine_global_init(s32 *restrict match, s32 *restrict gap_x,
			s32 *restrict gap_y, sequence_ptr_t seq1,
			sequence_ptr_t seq2)
{
	match[0] = 0;
	gap_x[0] = gap_y[0] = SCORE_MIN;

	const u64 len1 = seq1->length;
	const u64 len2 = seq2->length;
	const u64 cols = len1 + 1;
	const s32 gap_open = args_gap_open();
	const s32 gap_extend = args_gap_extend();

	UNROLL(8) for (u64 j = 1; j <= len1; j++)
	{
		gap_x[j] =
			max(match[j - 1] - gap_open, gap_x[j - 1] - gap_extend);
		match[j] = gap_x[j];
		gap_y[j] = SCORE_MIN;
	}

	UNROLL(8) for (u64 i = 1; i <= len2; i++)
	{
		const u64 idx = i * cols;
		gap_y[idx] = max(match[idx - cols] - gap_open,
				 gap_y[idx - cols] - gap_extend);
		match[idx] = gap_y[idx];
		gap_x[idx] = SCORE_MIN;
	}
}

void affine_global_fill(s32 *restrict match, s32 *restrict gap_x,
			s32 *restrict gap_y, const s32 *restrict seq1_indices,
			sequence_ptr_t seq1, sequence_ptr_t seq2)
{
	const u64 len1 = seq1->length;
	const u64 len2 = seq2->length;
	const u64 cols = len1 + 1;
	const s32 gap_open = args_gap_open();
	const s32 gap_extend = args_gap_extend();

	for (u64 i = 1; i <= len2; ++i) {
		const u64 row_offset = i * cols;
		const u64 prev_row_offset = (i - 1) * cols;
		const int c2_idx = SEQUENCE_LOOKUP[(uchar)seq2->letters[i - 1]];

		prefetch(&match[row_offset + PREFETCH_DISTANCE]);
		prefetch(&gap_x[row_offset + PREFETCH_DISTANCE]);
		prefetch(&gap_y[row_offset + PREFETCH_DISTANCE]);

		VECTORIZE for (u64 j = 1; j <= len1; j++)
		{
			const s32 similarity =
				SCORING_MATRIX[seq1_indices[j - 1]][c2_idx];
			const s32 diag_score =
				match[prev_row_offset + j - 1] + similarity;

			const s32 prev_match_x = match[row_offset + j - 1];
			const s32 prev_gap_x = gap_x[row_offset + j - 1];
			const s32 prev_match_y = match[prev_row_offset + j];
			const s32 prev_gap_y = gap_y[prev_row_offset + j];

			const s32 open_x = prev_match_x - gap_open;
			const s32 extend_x = prev_gap_x - gap_extend;
			gap_x[row_offset + j] = (open_x > extend_x) ? open_x :
								      extend_x;

			const s32 open_y = prev_match_y - gap_open;
			const s32 extend_y = prev_gap_y - gap_extend;
			gap_y[row_offset + j] = (open_y > extend_y) ? open_y :
								      extend_y;

			s32 best = diag_score;
			if (gap_x[row_offset + j] > best)
				best = gap_x[row_offset + j];
			if (gap_y[row_offset + j] > best)
				best = gap_y[row_offset + j];
			match[row_offset + j] = best;
		}
	}
}
