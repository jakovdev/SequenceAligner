#include "bio/method.h"

#include <args.h>
#include <print.h>
#include <strings.h>

#include "util/macros.h"

[[gnu::nonnull, gnu::noinline, gnu::hot]]
static shz align_ga(uhz l1, uhz l2, shz *restrict s1i, const u8 *restrict s2)
{
	if (LEN_BAD(l1) || LEN_BAD(l2) || SEQ_BAD(s2))
		unreachable_release();

	shz *restrict match = s1i + l1;
	shz *restrict gap_x = s1i + l1 + TABLE_SIZE;
	shz *restrict gap_y = s1i + l1 + TABLE_SIZE * 2;

	match[0] = 0;
	gap_x[0] = gap_y[0] = SCORE_MIN;

	for (uhz i = 1; i <= l1; i++) {
		gap_x[i] = max(match[i - 1] + GAP_OPN, gap_x[i - 1] + GAP_EXT);
		match[i] = gap_x[i];
		gap_y[i] = SCORE_MIN;
	}

	usz cols = l1 + 1;
	for (uhz i = 1; i <= l2; i++) {
		usz j = cols * i;
		gap_y[j] = max(match[j - cols] + GAP_OPN,
			       gap_y[j - cols] + GAP_EXT);
		match[j] = gap_y[j];
		gap_x[j] = SCORE_MIN;
	}

	for (uhz i = 1; i <= l2; ++i) {
		usz row = cols * i;
		usz row_prev = cols * (i - 1);
		shz c2 = SEQ_LUT[s2[i - 1]];

		for (uhz j = 1; j <= l1; j++) {
			shz similarity = SUB_MAT[s1i[j - 1]][c2];
			shz score_diag = match[row_prev + j - 1] + similarity;

			shz opn_x = match[row + j - 1] + GAP_OPN;
			shz ext_x = gap_x[row + j - 1] + GAP_EXT;
			shz opn_y = match[row_prev + j] + GAP_OPN;
			shz ext_y = gap_y[row_prev + j] + GAP_EXT;

			shz gap_x_curr = max(opn_x, ext_x);
			shz gap_y_curr = max(opn_y, ext_y);

			gap_x[row + j] = gap_x_curr;
			gap_y[row + j] = gap_y_curr;

			shz best = score_diag;
			best = max(gap_x_curr, best);
			best = max(gap_y_curr, best);
			match[row + j] = best;
		}
	}

	return match[cols * l2 + l1];
}

struct arg_callback validate_ga(void)
{
	if (GAP_OPN != GAP_EXT)
		return ARG_VALID();
	auto a = __start_methods;
	for (; a < __stop_methods; a++) {
		if (strcasecmp(a->name, "Needleman-Wunsch") == 0)
			break;
	}
	if (a == __stop_methods)
		return ARG_VALID();
	if (!print_Yn("Equal affine gaps found, switch to Needleman-Wunsch?"))
		return ARG_VALID();
	GAP_PEN = GAP_OPN;
	GAP_OPN = SCORE_MIN;
	GAP_EXT = SCORE_MIN;
	ALIGN = a;
	return ARG_VALID();
}

ALIGN_REGISTER("Gotoh", ga, GAP_AFFINE);
