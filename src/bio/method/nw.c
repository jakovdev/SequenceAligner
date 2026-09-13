#include "bio/method.h"
#include "util/macros.h"

[[gnu::nonnull, gnu::noinline, gnu::hot]]
static shz align_nw(uhz l1, uhz l2, shz *restrict s1i, const u8 *restrict s2)
{
	if (LEN_BAD(l1) || LEN_BAD(l2) || SEQ_BAD(s2))
		unreachable_release();

	shz *restrict table = s1i + l1;
	table[0] = 0;

	for (uhz i = 1; i <= l1; i++)
		table[i] = i * GAP_PEN;

	usz cols = l1 + 1;
	for (uhz i = 1; i <= l2; i++)
		table[cols * i] = i * GAP_PEN;

	for (uhz i = 1; i <= l2; ++i) {
		shz c2 = SEQ_LUT[s2[i - 1]];
		shz *restrict sub = SUB_MAT[c2];
		shz *restrict curr = table + cols * i;
		shz *restrict prev = curr - cols;
		shz left = curr[0];

		for (uhz j = 1; j <= l1; j++) {
			shz match = prev[j - 1] + sub[s1i[j - 1]];
			shz del = prev[j] + GAP_PEN;
			shz ins = left + GAP_PEN;

			shz val_max = match;
			val_max = max(del, val_max);
			val_max = max(ins, val_max);
			curr[j] = val_max;
			left = val_max;
		}
	}

	return table[cols * l2 + l1];
}

ALIGN_REGISTER("Needleman-Wunsch", nw, GAP_LINEAR);
