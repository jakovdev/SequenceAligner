#include "bio/align.h"
#include "util/macros.h"

[[gnu::nonnull, gnu::noinline, gnu::hot]]
static s32 align_nw(s32 len1, s32 len2, const uchar *restrict seq2,
		    const s32 *restrict ind, s32 *restrict table)
{
	if (LEN_BAD(len1) || LEN_BAD(len2) || SEQ_BAD(seq2))
		unreachable_release();

	s64 cols = len1 + 1;

	table[0] = 0;

	for (s32 i = 1; i <= len1; i++)
		table[i] = i * GAP_PEN;

	for (s32 i = 1; i <= len2; i++)
		table[cols * i] = i * GAP_PEN;

	for (s32 i = 1; i <= len2; ++i) {
		s32 c2 = SEQ_LUT[seq2[i - 1]];
		s32 *restrict sub = SUB_MAT[c2];
		s32 *restrict curr = table + cols * i;
		s32 *restrict prev = curr - cols;
		s32 left = curr[0];

		for (s32 j = 1; j <= len1; j++) {
			s32 match = prev[j - 1] + sub[ind[j - 1]];
			s32 del = prev[j] + GAP_PEN;
			s32 ins = left + GAP_PEN;

			s32 val_max = match;
			val_max = max(del, val_max);
			val_max = max(ins, val_max);
			curr[j] = val_max;
			left = val_max;
		}
	}

	return table[cols * len2 + len1];
}

ALIGN_KERNEL(kernel_nw);

ALIGN_REGISTER(nw) = {
	.ALIGN_ALIASES("Needleman-Wunsch", "nw"),
	.method = align_nw,
	.kernel = kernel_nw,
	.gap = GAP_LINEAR,
};
