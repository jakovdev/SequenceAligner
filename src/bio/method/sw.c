#include "bio/alignment.h"
#include "util/macros.h"

[[gnu::nonnull, gnu::noinline, gnu::hot]]
static s32 align_sw(s32 len1, s32 len2, seq seq2, const s32 *restrict ind,
		    s32 *restrict table)
{
	if (LEN_BAD(len1) || LEN_BAD(len2) || SEQ_BAD(seq2))
		unreachable_release();

	s64 cols = len1 + 1;

	extern size_t TABLE_SIZE;
	s32 *restrict match = table;
	s32 *restrict gap_x = table + TABLE_SIZE;
	s32 *restrict gap_y = table + TABLE_SIZE * 2;

	match[0] = 0;
	gap_x[0] = gap_y[0] = SCORE_MIN;

	for (s32 i = 1; i <= len1; i++) {
		match[i] = 0;
		gap_x[i] = gap_y[i] = SCORE_MIN;
	}

	for (s32 i = 1; i <= len2; i++) {
		s64 j = cols * i;
		match[j] = 0;
		gap_x[j] = gap_y[j] = SCORE_MIN;
	}

	s32 score = 0;
	for (s32 i = 1; i <= len2; ++i) {
		s64 row = cols * i;
		s64 row_prev = cols * (i - 1);
		s32 c2 = SEQ_LUT[(uchar)seq2[i - 1]];

		for (s32 j = 1; j <= len1; j++) {
			s32 similarity = SUB_MAT[ind[j - 1]][c2];
			s32 score_diag = match[row_prev + j - 1] + similarity;

			s32 opn_x = match[row + j - 1] + GAP_OPN;
			s32 ext_x = gap_x[row + j - 1] + GAP_EXT;
			s32 opn_y = match[row_prev + j] + GAP_OPN;
			s32 ext_y = gap_y[row_prev + j] + GAP_EXT;

			s32 gap_x_curr = max(opn_x, ext_x);
			s32 gap_y_curr = max(opn_y, ext_y);

			gap_x[row + j] = gap_x_curr;
			gap_y[row + j] = gap_y_curr;

			s32 best = max(score_diag, 0);
			best = max(gap_x_curr, best);
			best = max(gap_y_curr, best);
			match[row + j] = best;
			score = max(score, best);
		}
	}

	return score;
}

ALIGN_KERNEL(kernel_sw);

ALIGN_REGISTER(sw) = {
	.ALIGN_ALIASES("Smith-Waterman", "sw"),
	.method = align_sw,
	.kernel = kernel_sw,
	.gap = GAP_AFFINE,
};
