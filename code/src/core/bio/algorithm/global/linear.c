#include "core/bio/algorithm/global/linear.h"

#include "core/app/args.h"
#include "core/bio/score/scoring.h"
#include "system/compiler.h"
#include "system/simd.h"

void linear_global_init(s32 *restrict matrix, sequence_ptr_t seq1,
			sequence_ptr_t seq2)
{
	const u64 len1 = seq1->length;
	const u64 len2 = seq2->length;
	const u64 cols = len1 + 1;
	const s32 gap_penalty = args_gap_penalty();

	matrix[0] = 0;
	VECTORIZE UNROLL(8) for (u64 j = 1; j <= len1; j++)
	{
		matrix[j] = (s32)j * (-gap_penalty);
	}

	VECTORIZE UNROLL(8) for (u64 i = 1; i <= len2; i++)
	{
		matrix[i * cols] = (s32)i * (-gap_penalty);
	}
}

void linear_global_fill(s32 *restrict matrix, const s32 *restrict seq1_indices,
			sequence_ptr_t seq1, sequence_ptr_t seq2)
{
	const u64 len1 = seq1->length;
	const u64 len2 = seq2->length;
	const u64 cols = len1 + 1;
	const s32 gap_penalty = args_gap_penalty();
	for (u64 i = 1; i <= len2; ++i) {
		const u64 row_offset = i * cols;
		const u64 prev_row_offset = (i - 1) * cols;
		const int c2_idx = SEQUENCE_LOOKUP[(uchar)seq2->letters[i - 1]];

		prefetch(&matrix[row_offset + PREFETCH_DISTANCE]);

		UNROLL(4) for (u64 j = 1; j <= len1; j++)
		{
			const s32 match =
				matrix[prev_row_offset + j - 1] +
				SCORING_MATRIX[seq1_indices[j - 1]][c2_idx];
			const s32 del =
				matrix[prev_row_offset + j] + (-gap_penalty);
			const s32 insert =
				matrix[row_offset + j - 1] + (-gap_penalty);
			matrix[row_offset + j] =
				match > del ?
					(match > insert ? match : insert) :
					(del > insert ? del : insert);
		}
	}
}

#if USE_SIMD == 1

void simd_linear_global_row_init(s32 *restrict matrix, sequence_ptr_t seq1)
{
	const u64 len1 = seq1->length;
	const s32 gap_penalty = args_gap_penalty();

	veci_t indices = g_first_row_indices;
	veci_t gap_penalty_vec = set1_epi32(-gap_penalty);
	for (u64 j = 1; j <= len1; j += NUM_ELEMS) {
		u64 remaining = len1 + 1 - j;
		if (remaining >= NUM_ELEMS) {
			veci_t values = mullo_epi32(indices, gap_penalty_vec);
			storeu((veci_t *)&matrix[j], values);
		} else {
			for (u64 k = 0; k < remaining; k++)
				matrix[j + k] = (s32)(j + k) * (-gap_penalty);
		}

		indices = add_epi32(indices, set1_epi32(NUM_ELEMS));
	}
}

#endif
