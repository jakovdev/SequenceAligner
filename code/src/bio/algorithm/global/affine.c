#include "bio/algorithm/global/affine.h"

#include "bio/score/scoring.h"
#include "bio/types.h"
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
	const s32 gap_open = arg_gap_open();
	const s32 gap_ext = arg_gap_ext();

	UNROLL(8)
	for (u64 j = 1; j <= len1; j++) {
		gap_x[j] = max(match[j - 1] + gap_open, gap_x[j - 1] + gap_ext);
		match[j] = gap_x[j];
		gap_y[j] = SCORE_MIN;
	}

	UNROLL(8)
	for (u64 i = 1; i <= len2; i++) {
		const u64 idx = i * cols;
		gap_y[idx] = max(match[idx - cols] + gap_open,
				 gap_y[idx - cols] + gap_ext);
		match[idx] = gap_y[idx];
		gap_x[idx] = SCORE_MIN;
	}
}

void affine_global_fill(s32 *restrict match, s32 *restrict gap_x,
			s32 *restrict gap_y, const s32 *restrict seq1_i,
			sequence_ptr_t seq1, sequence_ptr_t seq2)
{
	const u64 len1 = seq1->length;
	const u64 len2 = seq2->length;
	const u64 cols = len1 + 1;
	const s32 gap_open = arg_gap_open();
	const s32 gap_ext = arg_gap_ext();

	for (u64 i = 1; i <= len2; ++i) {
		const u64 row = i * cols;
		const u64 p_row = (i - 1) * cols;
		const s32 c2_idx = SEQ_LUP[(uchar)seq2->letters[i - 1]];

		prefetch(&match[row + PREFETCH_DISTANCE]);
		prefetch(&gap_x[row + PREFETCH_DISTANCE]);
		prefetch(&gap_y[row + PREFETCH_DISTANCE]);

		VECTORIZE
		for (u64 j = 1; j <= len1; j++) {
			const s32 similarity = SUB_MAT[seq1_i[j - 1]][c2_idx];
			const s32 d_score = match[p_row + j - 1] + similarity;

			const s32 p_match_x = match[row + j - 1];
			const s32 p_gap_x = gap_x[row + j - 1];
			const s32 p_match_y = match[p_row + j];
			const s32 p_gap_y = gap_y[p_row + j];

			const s32 open_x = p_match_x + gap_open;
			const s32 extend_x = p_gap_x + gap_ext;
			gap_x[row + j] = (open_x > extend_x) ? open_x :
							       extend_x;

			const s32 open_y = p_match_y + gap_open;
			const s32 extend_y = p_gap_y + gap_ext;
			gap_y[row + j] = (open_y > extend_y) ? open_y :
							       extend_y;

			s32 best = d_score;
			if (gap_x[row + j] > best)
				best = gap_x[row + j];
			if (gap_y[row + j] > best)
				best = gap_y[row + j];
			match[row + j] = best;
		}
	}
}
