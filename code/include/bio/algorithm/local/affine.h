#pragma once
#ifndef BIO_ALGORITHM_LOCAL_AFFINE_H
#define BIO_ALGORITHM_LOCAL_AFFINE_H

#include "bio/types.h"
#include "system/simd.h"
#include "system/types.h"

void affine_local_init(s32 *restrict match, s32 *restrict gap_x,
		       s32 *restrict gap_y, sequence_ptr_t seq1,
		       sequence_ptr_t seq2);

s32 affine_local_fill(s32 *restrict match, s32 *restrict gap_x,
		      s32 *restrict gap_y, const s32 *restrict seq1_i,
		      sequence_ptr_t seq1, sequence_ptr_t seq2);

#if USE_SIMD == 1
void affine_local_init_simd(s32 *restrict match, s32 *restrict gap_x,
			    s32 *restrict gap_y, sequence_ptr_t seq1,
			    sequence_ptr_t seq2);
#endif

#endif /* BIO_ALGORITHM_LOCAL_AFFINE_H */
