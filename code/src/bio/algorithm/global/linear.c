#include "bio/algorithm/global/linear.h"

#include "app/args.h"
#include "bio/score/scoring.h"
#include "system/compiler.h"
#include "system/simd.h"

void linear_global_init(s32 *restrict matrix, sequence_ptr_t seq1,
			sequence_ptr_t seq2)
{
	const u64 len1 = seq1->length;
	const u64 len2 = seq2->length;
	const u64 cols = len1 + 1;
	const s32 gap_pen = args_gap_pen();

	matrix[0] = 0;

	VECTORIZE
	UNROLL(8)
	for (u64 j = 1; j <= len1; j++)
		matrix[j] = (s32)j * (-gap_pen);

	VECTORIZE
	UNROLL(8)
	for (u64 i = 1; i <= len2; i++)
		matrix[i * cols] = (s32)i * (-gap_pen);
}

void linear_global_fill(s32 *restrict matrix, const s32 *restrict seq1_i,
			sequence_ptr_t seq1, sequence_ptr_t seq2)
{
	const u64 len1 = seq1->length;
	const u64 len2 = seq2->length;
	const u64 cols = len1 + 1;
	const s32 gap_pen = args_gap_pen();
	for (u64 i = 1; i <= len2; ++i) {
		const u64 row = i * cols;
		const u64 p_row = (i - 1) * cols;
		const s32 c2_idx = SEQ_LUP[(uchar)seq2->letters[i - 1]];

		prefetch(&matrix[row + PREFETCH_DISTANCE]);

		UNROLL(4)
		for (u64 j = 1; j <= len1; j++) {
			const s32 match = matrix[p_row + j - 1] +
					  SUB_MAT[seq1_i[j - 1]][c2_idx];
			const s32 del = matrix[p_row + j] + (-gap_pen);
			const s32 insert = matrix[row + j - 1] + (-gap_pen);
			matrix[row + j] =
				match > del ?
					(match > insert ? match : insert) :
					(del > insert ? del : insert);
		}
	}
}
