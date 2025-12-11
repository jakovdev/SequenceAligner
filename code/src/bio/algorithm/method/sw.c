#include "bio/algorithm/method/sw.h"

#include "bio/algorithm/local/affine.h"
#include "system/compiler.h"

s32 align_sw(sequence_ptr_t seq1, sequence_ptr_t seq2)
{
	if (SEQ_INVALID(seq1) || SEQ_INVALID(seq2))
		unreachable_release();

	affine_local_init(seq1, seq2);
	return affine_local_fill(seq1, seq2);
}
