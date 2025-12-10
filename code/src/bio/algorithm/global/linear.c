#include "bio/algorithm/global/linear.h"

#include "bio/algorithm/indices.h"
#include "bio/algorithm/matrix.h"
#include "system/compiler.h"

void linear_global_init(sequence_ptr_t seq1, sequence_ptr_t seq2)
{
	if (SEQ_INVALID(seq1) || SEQ_INVALID(seq2) || !g_matrix)
		unreachable();

	const s32 len1 = seq1->length;
	const s32 len2 = seq2->length;
	const s64 cols = len1 + 1;
	const s32 gap_pen = arg_gap_pen();

	g_matrix[0] = 0;

	for (s32 j = 1; j <= len1; j++)
		g_matrix[j] = j * gap_pen;

	for (s32 i = 1; i <= len2; i++)
		g_matrix[cols * i] = i * gap_pen;
}

s32 linear_global_fill(sequence_ptr_t seq1, sequence_ptr_t seq2)
{
	if (SEQ_INVALID(seq1) || SEQ_INVALID(seq2) || !g_matrix || !g_seq1_i)
		unreachable();

	const s32 len1 = seq1->length;
	const s32 len2 = seq2->length;
	const s64 cols = len1 + 1;
	const s32 gap_pen = arg_gap_pen();

	for (s32 i = 1; i <= len2; ++i) {
		const s64 row = cols * i;
		const s64 p_row = cols * (i - 1);
		const s32 c2_idx = SEQ_LUP[(uchar)seq2->letters[i - 1]];

		for (s32 j = 1; j <= len1; j++) {
			const s32 match = g_matrix[p_row + j - 1] +
					  SUB_MAT[g_seq1_i[j - 1]][c2_idx];
			const s32 del = g_matrix[p_row + j] + gap_pen;
			const s32 insert = g_matrix[row + j - 1] + gap_pen;
			s32 max_val = match;
			max_val = del > max_val ? del : max_val;
			max_val = insert > max_val ? insert : max_val;
			g_matrix[row + j] = max_val;
		}
	}

	return g_matrix[(s64)len2 * (len1 + 1) + len1];
}
