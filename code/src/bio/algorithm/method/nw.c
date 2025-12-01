#include "bio/algorithm/method/nw.h"

#include "bio/algorithm/global/linear.h"

s32 align_nw(sequence_ptr_t seq1, sequence_ptr_t seq2)
{
	linear_global_init(seq1, seq2);
	return linear_global_fill(seq1, seq2);
}
