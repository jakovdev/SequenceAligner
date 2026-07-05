#include "bio/method.h"
#include "util/macros.h"

[[gnu::nonnull, gnu::noinline, gnu::hot]]
static s32 align_nw(s32 l1, s32 l2, s32 *restrict s1i, const u8 *restrict s2)
{
	if (LEN_BAD(l1) || LEN_BAD(l2) || SEQ_BAD(s2))
		unreachable_release();

	s32 *restrict table = s1i + l1;
	table[0] = 0;

	for (s32 i = 1; i <= l1; i++)
		table[i] = i * GAP_PEN;

	s64 cols = l1 + 1;
	for (s32 i = 1; i <= l2; i++)
		table[cols * i] = i * GAP_PEN;

	for (s32 i = 1; i <= l2; ++i) {
		s32 c2 = SEQ_LUT[s2[i - 1]];
		s32 *restrict sub = SUB_MAT[c2];
		s32 *restrict curr = table + cols * i;
		s32 *restrict prev = curr - cols;
		s32 left = curr[0];

		for (s32 j = 1; j <= l1; j++) {
			s32 match = prev[j - 1] + sub[s1i[j - 1]];
			s32 del = prev[j] + GAP_PEN;
			s32 ins = left + GAP_PEN;

			s32 val_max = match;
			val_max = max(del, val_max);
			val_max = max(ins, val_max);
			curr[j] = val_max;
			left = val_max;
		}
	}

	return table[cols * l2 + l1];
}

ALIGN_REGISTER("Needleman-Wunsch", nw, GAP_LINEAR);
