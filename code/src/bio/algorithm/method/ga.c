#include "bio/algorithm/method/ga.h"

#include "bio/algorithm/global/affine.h"

s32 align_ga(sequence_ptr_t seq1, sequence_ptr_t seq2)
{
	affine_global_init(seq1, seq2);
	return affine_global_fill(seq1, seq2);
}
