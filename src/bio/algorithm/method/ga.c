#include "bio/algorithm/method/ga.h"

#include "bio/algorithm/global/affine.h"

s32 align_ga(SEQUENCE_PTR_T(seq1), SEQUENCE_PTR_T(seq2))
{
	affine_global_init(seq1, seq2);
	return affine_global_fill(seq1, seq2);
}
