#include "bio/alignment.h"
#include "util/macros.h"

extern size_t TABLE_SIZE;

[[gnu::nonnull, gnu::noinline, gnu::hot]]
static s32 align_sw(seq_ptr seq1, seq_ptr seq2, s32 *restrict table,
		    s32 *restrict ind)
{
	if (SEQ_BAD(seq1) || SEQ_BAD(seq2))
		unreachable_release();

	const s32 len1 = seq1->length;
	const s32 len2 = seq2->length;
	const s64 cols = len1 + 1;

	s32 *restrict match = table;
	s32 *restrict gap_x = table + TABLE_SIZE;
	s32 *restrict gap_y = table + 2 * TABLE_SIZE;

	match[0] = 0;
	gap_x[0] = gap_y[0] = SCORE_MIN;

	for (s32 j = 1; j <= len1; j++) {
		match[j] = 0;
		gap_x[j] = gap_y[j] = SCORE_MIN;
	}

	for (s32 i = 1; i <= len2; i++) {
		s64 idx = cols * i;
		match[idx] = 0;
		gap_x[idx] = gap_y[idx] = SCORE_MIN;
	}

	s32 score = 0;
	for (s32 i = 1; i <= len2; ++i) {
		const s64 row = cols * i;
		const s64 p_row = cols * (i - 1);
		const s32 c2_idx = SEQ_LUT[(uchar)seq2->letters[i - 1]];

		for (s32 j = 1; j <= len1; j++) {
			const s32 similarity = SUB_MAT[ind[j - 1]][c2_idx];
			const s32 d_score = match[p_row + j - 1] + similarity;

			const s32 p_match_x = match[row + j - 1];
			const s32 p_gap_x = gap_x[row + j - 1];
			const s32 p_match_y = match[p_row + j];
			const s32 p_gap_y = gap_y[p_row + j];

			const s32 opn_x = p_match_x + GAP_OPN;
			const s32 ext_x = p_gap_x + GAP_EXT;
			const s32 opn_y = p_match_y + GAP_OPN;
			const s32 ext_y = p_gap_y + GAP_EXT;

			const s32 c_gap_x = opn_x > ext_x ? opn_x : ext_x;
			const s32 c_gap_y = opn_y > ext_y ? opn_y : ext_y;

			gap_x[row + j] = c_gap_x;
			gap_y[row + j] = c_gap_y;

			s32 best = 0;
			best = d_score > best ? d_score : best;
			best = c_gap_x > best ? c_gap_x : best;
			best = c_gap_y > best ? c_gap_y : best;
			match[row + j] = best;
			if (best > score)
				score = best;
		}
	}

	return score;
}
ALIGN_METHOD(ALIGN_SW, align_sw, GAP_AFFINE, "Smith-Waterman", "sw", "smith",
	     "waterman")
