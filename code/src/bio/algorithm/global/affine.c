#include "bio/algorithm/global/affine.h"

#include "bio/algorithm/indices.h"
#include "bio/algorithm/matrix.h"
#include "bio/score/matrices.h"
#include "system/compiler.h"
#include "system/os.h"

void affine_global_init(SEQUENCE_PTR_T(seq1), SEQUENCE_PTR_T(seq2))
{
	if (SEQ_BAD(seq1) || SEQ_BAD(seq2) || !MATCH || !GAP_X || !GAP_Y)
		unreachable_release();

	const s32 len1 = seq1->length;
	const s32 len2 = seq2->length;
	const s64 cols = len1 + 1;

	MATCH[0] = 0;
	GAP_X[0] = GAP_Y[0] = SCORE_MIN;

	for (s32 j = 1; j <= len1; j++) {
		GAP_X[j] = max(MATCH[j - 1] + GAP_OPEN, GAP_X[j - 1] + GAP_EXT);
		MATCH[j] = GAP_X[j];
		GAP_Y[j] = SCORE_MIN;
	}

	for (s32 i = 1; i <= len2; i++) {
		const s64 idx = cols * i;
		GAP_Y[idx] = max(MATCH[idx - cols] + GAP_OPEN,
				 GAP_Y[idx - cols] + GAP_EXT);
		MATCH[idx] = GAP_Y[idx];
		GAP_X[idx] = SCORE_MIN;
	}
}

s32 affine_global_fill(SEQUENCE_PTR_T(seq1), SEQUENCE_PTR_T(seq2))
{
	if (SEQ_BAD(seq1) || SEQ_BAD(seq2) || !MATCH || !GAP_X || !GAP_Y ||
	    !SEQ1I)
		unreachable_release();

	const s32 len1 = seq1->length;
	const s32 len2 = seq2->length;
	const s64 cols = len1 + 1;

	for (s32 i = 1; i <= len2; ++i) {
		const s64 row = cols * i;
		const s64 p_row = cols * (i - 1);
		const s32 c2_idx = SEQ_LUT[(uchar)seq2->letters[i - 1]];

		for (s32 j = 1; j <= len1; j++) {
			const s32 similarity = SUB_MAT[SEQ1I[j - 1]][c2_idx];
			const s32 d_score = MATCH[p_row + j - 1] + similarity;

			const s32 p_match_x = MATCH[row + j - 1];
			const s32 p_gap_x = GAP_X[row + j - 1];
			const s32 p_match_y = MATCH[p_row + j];
			const s32 p_gap_y = GAP_Y[p_row + j];

			const s32 open_x = p_match_x + GAP_OPEN;
			const s32 extend_x = p_gap_x + GAP_EXT;
			const s32 open_y = p_match_y + GAP_OPEN;
			const s32 extend_y = p_gap_y + GAP_EXT;

			const s32 gap_x = open_x > extend_x ? open_x : extend_x;
			const s32 gap_y = open_y > extend_y ? open_y : extend_y;

			GAP_X[row + j] = gap_x;
			GAP_Y[row + j] = gap_y;

			s32 best = d_score;
			best = gap_x > best ? gap_x : best;
			best = gap_y > best ? gap_y : best;
			MATCH[row + j] = best;
		}
	}

	return MATCH[(s64)len2 * (len1 + 1) + len1];
}
