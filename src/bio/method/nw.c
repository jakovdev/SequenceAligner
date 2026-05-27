#include "bio/alignment.h"
#include "bio/sequence.h"
#include "util/macros.h"

[[gnu::nonnull, gnu::noinline, gnu::hot]]
static s32 align_nw(const struct sequence *restrict seq1,
		    const struct sequence *restrict seq2,
		    const s32 *restrict ind, s32 *restrict table)
{
	if (SEQ_BAD(seq1) || SEQ_BAD(seq2))
		unreachable_release();

	s32 len1 = seq1->length;
	s32 len2 = seq2->length;
	s64 cols = len1 + 1;

	table[0] = 0;

	for (s32 i = 1; i <= len1; i++)
		table[i] = i * GAP_PEN;

	for (s32 i = 1; i <= len2; i++)
		table[cols * i] = i * GAP_PEN;

	for (s32 i = 1; i <= len2; ++i) {
		s32 c2 = SEQ_LUT[(uchar)seq2->letters[i - 1]];
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

	return table[(s64)len2 * cols + len1];
}
ALIGN_METHOD(ALIGN_NW, align_nw, GAP_LINEAR, "Needleman-Wunsch", "nw",
	     "needleman", "wunsch")
