#include "bio/algorithm/method/ga.h"

#include "bio/algorithm/global/affine.h"
#include "bio/algorithm/indices.h"
#include "bio/algorithm/matrix.h"
#include "util/print.h"
#include "system/memory.h"

s32 align_ga(sequence_ptr_t seq1, sequence_ptr_t seq2)
{
	const u64 len1 = seq1->length;
	const u64 len2 = seq2->length;
	const size_t mat_bytes = MATRICES_3X_BYTES(len1, len2);
	const size_t mat_size = MATRICES_3X_SIZE_S(len1, len2, mat_bytes);
#ifdef _MSC_VER
	s32 *stack_matrix = ALLOCA(stack_matrix, mat_size);
#else
	s32 stack_matrix[mat_size];
#endif
	s32 *restrict matrix = matrix_alloc(stack_matrix, mat_bytes);
	s32 *restrict match = matrix;
	s32 *restrict gap_x = matrix + MATRIX_SIZE(len1, len2);
	s32 *restrict gap_y = matrix + 2 * MATRIX_SIZE(len1, len2);
	affine_global_init(match, gap_x, gap_y, seq1, seq2);

	s32 *restrict seq1_i = { 0 };
	bool is_stack = false;
	if (len1 > MAX_STACK_SEQUENCE_LENGTH) {
		seq1_i = MALLOC(seq1_i, len1);
		if (UNLIKELY(!seq1_i)) {
			perror_context("SEQALIGN - GA");
			perror("Failed to allocate memory for sequence indices");
			exit(EXIT_FAILURE);
		}

		goto no_stack;
	}

	seq1_i = ALLOCA(seq1_i, len1);
	is_stack = true;

no_stack:
	seq_indices_precompute(seq1_i, seq1);
	affine_global_fill(match, gap_x, gap_y, seq1_i, seq1, seq2);
	const s32 score = match[len2 * (len1 + 1) + len1];
	seq_indices_free(seq1_i, is_stack);
	matrix_free(matrix, stack_matrix);
	return score;
}
