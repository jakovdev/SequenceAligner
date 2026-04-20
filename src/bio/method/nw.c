#include "bio/alignment.h"
#include "util/macros.h"

static s32 align_nw(seq_ptr seq1, seq_ptr seq2, s32 *restrict table,
		    s32 *restrict ind)
{
	if (SEQ_BAD(seq1) || SEQ_BAD(seq2))
		unreachable_release();

	const s32 len1 = seq1->length;
	const s32 len2 = seq2->length;
	const s64 cols = (s64)len1 + 1;

	table[0] = 0;

	for (s32 j = 1; j <= len1; j++)
		table[j] = j * GAP_PEN;

	for (s32 i = 1; i <= len2; i++)
		table[cols * i] = i * GAP_PEN;

	for (s32 i = 1; i <= len2; ++i) {
		const s32 c2_idx = SEQ_LUT[(uchar)seq2->letters[i - 1]];
		const s32 *restrict sub_row = SUB_MAT[c2_idx];
		s32 *restrict curr = table + cols * i;
		const s32 *restrict prev = curr - cols;
		s32 left = curr[0];

		for (s32 j = 1; j <= len1; j++) {
			const s32 match = prev[j - 1] + sub_row[ind[j - 1]];
			const s32 del = prev[j] + GAP_PEN;
			const s32 ins = left + GAP_PEN;

			s32 max_val = match;
			max_val = del > max_val ? del : max_val;
			max_val = ins > max_val ? ins : max_val;
			curr[j] = max_val;
			left = max_val;
		}
	}

	return table[(s64)len2 * cols + len1];
}
ALIGN_METHOD(ALIGN_NW, align_nw, GAP_LINEAR, "Needleman-Wunsch", "nw",
	     "needleman", "wunsch");
