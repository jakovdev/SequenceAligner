#include "bio/algorithm/global/linear.h"

#include "bio/algorithm/indices.h"
#include "bio/algorithm/matrix.h"
#include "system/compiler.h"
#include "system/simd.h"

void linear_global_init(sequence_ptr_t seq1, sequence_ptr_t seq2)
{
	const u64 len1 = seq1->length;
	const u64 len2 = seq2->length;
	const u64 cols = len1 + 1;
	const s32 gap_pen = arg_gap_pen();

	g_matrix[0] = 0;

	VECTORIZE
	UNROLL(8)
	for (u64 j = 1; j <= len1; j++)
		g_matrix[j] = (s32)j * gap_pen;

	VECTORIZE
	UNROLL(8)
	for (u64 i = 1; i <= len2; i++)
		g_matrix[i * cols] = (s32)i * gap_pen;
}

s32 linear_global_fill(sequence_ptr_t seq1, sequence_ptr_t seq2)
{
	const u64 len1 = seq1->length;
	const u64 len2 = seq2->length;
	const u64 cols = len1 + 1;
	const s32 gap_pen = arg_gap_pen();
	for (u64 i = 1; i <= len2; ++i) {
		const u64 row = i * cols;
		const u64 p_row = (i - 1) * cols;
		const s32 c2_idx = SEQ_LUP[(uchar)seq2->letters[i - 1]];

		prefetch(&g_matrix[row + PREFETCH_DISTANCE]);

		UNROLL(4)
		for (u64 j = 1; j <= len1; j++) {
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

	return g_matrix[len2 * (len1 + 1) + len1];
}
