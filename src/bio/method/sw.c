#include "bio/method.h"
#include "util/macros.h"

[[gnu::nonnull, gnu::noinline, gnu::hot]]
static s32 align_sw(s32 l1, s32 l2, s32 *restrict s1i, const u8 *restrict s2)
{
	if (LEN_BAD(l1) || LEN_BAD(l2) || SEQ_BAD(s2))
		unreachable_release();

	s32 *restrict match = s1i + l1;
	s32 *restrict gap_x = s1i + l1 + TABLE_SIZE;
	s32 *restrict gap_y = s1i + l1 + TABLE_SIZE * 2;

	match[0] = 0;
	gap_x[0] = gap_y[0] = SCORE_MIN;

	for (s32 i = 1; i <= l1; i++) {
		match[i] = 0;
		gap_x[i] = gap_y[i] = SCORE_MIN;
	}

	s64 cols = l1 + 1;
	for (s32 i = 1; i <= l2; i++) {
		s64 j = cols * i;
		match[j] = 0;
		gap_x[j] = gap_y[j] = SCORE_MIN;
	}

	s32 score = 0;
	for (s32 i = 1; i <= l2; ++i) {
		s64 row = cols * i;
		s64 row_prev = cols * (i - 1);
		s32 c2 = SEQ_LUT[s2[i - 1]];

		for (s32 j = 1; j <= l1; j++) {
			s32 similarity = SUB_MAT[s1i[j - 1]][c2];
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

ALIGN_REGISTER("Smith-Waterman", sw, GAP_AFFINE);
