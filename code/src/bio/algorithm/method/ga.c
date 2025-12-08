#include "bio/algorithm/method/ga.h"

#include "bio/algorithm/global/affine.h"
#include "system/compiler.h"

s32 align_ga(sequence_ptr_t seq1, sequence_ptr_t seq2)
{
	if (SEQ_INVALID(seq1) || SEQ_INVALID(seq2))
		unreachable();

	affine_global_init(seq1, seq2);
	return affine_global_fill(seq1, seq2);
}
