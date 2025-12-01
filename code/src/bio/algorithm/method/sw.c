#include "bio/algorithm/method/sw.h"

#include "bio/algorithm/local/affine.h"

s32 align_sw(sequence_ptr_t seq1, sequence_ptr_t seq2)
{
	affine_local_init(seq1, seq2);
	return affine_local_fill(seq1, seq2);
}
