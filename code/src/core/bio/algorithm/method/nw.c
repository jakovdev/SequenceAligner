#include "core/bio/algorithm/method/nw.h"

#include "core/bio/algorithm/global/linear.h"
#include "core/bio/algorithm/indices.h"
#include "core/bio/algorithm/matrix.h"
#include "util/print.h"
#include "system/memory.h"

s32 align_nw(sequence_ptr_t seq1, sequence_ptr_t seq2)
{
	const u64 len1 = seq1->length;
	const u64 len2 = seq2->length;
	const size_t mat_bytes = MATRIX_BYTES(len1, len2);
	const size_t mat_size = MATRIX_SIZE_S(len1, len2, mat_bytes);
#ifdef _MSC_VER
	s32 *stack_matrix = ALLOCA(stack_matrix, mat_size);
#else
	s32 stack_matrix[mat_size];
#endif
	s32 *restrict matrix = matrix_alloc(stack_matrix, mat_bytes);
	linear_global_init(matrix, seq1, seq2);
	s32 *restrict seq1_i = { 0 };
	bool is_stack = false;
	if (len1 > MAX_STACK_SEQUENCE_LENGTH) {
		seq1_i = MALLOC(seq1_i, len1);
		if (UNLIKELY(!seq1_i)) {
			print_error_context("SEQALIGN - NW");
			print(M_NONE, ERR
			      "Failed to allocate memory for sequence indices");
			exit(EXIT_FAILURE);
		}

		goto no_stack;
	}

	seq1_i = ALLOCA(seq1_i, len1);
	is_stack = true;

no_stack:
	seq_indices_precompute(seq1_i, seq1);
	linear_global_fill(matrix, seq1_i, seq1, seq2);
	const s32 score = matrix[len2 * (len1 + 1) + len1];
	seq_indices_free(seq1_i, is_stack);
	matrix_free(matrix, stack_matrix);
	return score;
}
