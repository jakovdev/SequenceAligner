#include "bio/align.h"

#include <args.h>
#include <print.h>
#include <string.h>

#include "util/macros.h"

[[gnu::nonnull, gnu::noinline, gnu::hot]]
static s32 align_ga(s32 len1, s32 len2, seq seq2, const s32 *restrict ind,
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
		gap_x[i] = max(match[i - 1] + GAP_OPN, gap_x[i - 1] + GAP_EXT);
		match[i] = gap_x[i];
		gap_y[i] = SCORE_MIN;
	}

	for (s32 i = 1; i <= len2; i++) {
		s64 j = cols * i;
		gap_y[j] = max(match[j - cols] + GAP_OPN,
			       gap_y[j - cols] + GAP_EXT);
		match[j] = gap_y[j];
		gap_x[j] = SCORE_MIN;
	}

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

			s32 best = score_diag;
			best = max(gap_x_curr, best);
			best = max(gap_y_curr, best);
			match[row + j] = best;
		}
	}

	return match[(s64)len2 * (len1 + 1) + len1];
}

static struct arg_callback validate_ga(void)
{
	if (GAP_OPN != GAP_EXT)
		return ARG_VALID();
	auto a = __start_aligns;
	for (; a < __stop_aligns; a++) {
		if (strcasecmp(*a->aliases, "Needleman-Wunsch") == 0)
			break;
	}
	if (a == __stop_aligns)
		return ARG_VALID();
	if (!print_Yn("Equal affine gaps found, switch to Needleman-Wunsch?"))
		return ARG_VALID();
	GAP_PEN = GAP_OPN;
	GAP_OPN = SCORE_MIN;
	GAP_EXT = SCORE_MIN;
	ALIGN = a;
	return ARG_VALID();
}

ALIGN_KERNEL(kernel_ga);

ALIGN_REGISTER(ga) = {
	.ALIGN_ALIASES("Gotoh", "ga"),
	.method = align_ga,
	.validate = validate_ga,
	.kernel = kernel_ga,
	.gap = GAP_AFFINE,
};
