#include "bio/algorithm/global/linear.h"

#include "bio/algorithm/indices.h"
#include "bio/algorithm/matrix.h"
#include "bio/score/matrices.h"
#include "system/compiler.h"

void linear_global_init(SEQUENCE_PTR_T(seq1), SEQUENCE_PTR_T(seq2))
{
	if (SEQ_BAD(seq1) || SEQ_BAD(seq2) || !MATRIX)
		unreachable_release();

	const s32 len1 = seq1->length;
	const s32 len2 = seq2->length;
	const s64 cols = len1 + 1;

	MATRIX[0] = 0;

	for (s32 j = 1; j <= len1; j++)
		MATRIX[j] = j * GAP_PEN;

	for (s32 i = 1; i <= len2; i++)
		MATRIX[cols * i] = i * GAP_PEN;
}

s32 linear_global_fill(SEQUENCE_PTR_T(seq1), SEQUENCE_PTR_T(seq2))
{
	if (SEQ_BAD(seq1) || SEQ_BAD(seq2) || !MATRIX || !SEQ1I)
		unreachable_release();

	const s32 len1 = seq1->length;
	const s32 len2 = seq2->length;
	const s64 cols = len1 + 1;

	for (s32 i = 1; i <= len2; ++i) {
		const s64 row = cols * i;
		const s64 p_row = cols * (i - 1);
		const s32 c2_idx = SEQ_LUT[(uchar)seq2->letters[i - 1]];

		for (s32 j = 1; j <= len1; j++) {
			const s32 match = MATRIX[p_row + j - 1] +
					  SUB_MAT[SEQ1I[j - 1]][c2_idx];
			const s32 del = MATRIX[p_row + j] + GAP_PEN;
			const s32 insert = MATRIX[row + j - 1] + GAP_PEN;
			s32 max_val = match;
			max_val = del > max_val ? del : max_val;
			max_val = insert > max_val ? insert : max_val;
			MATRIX[row + j] = max_val;
		}
	}

	return MATRIX[(s64)len2 * (len1 + 1) + len1];
}
