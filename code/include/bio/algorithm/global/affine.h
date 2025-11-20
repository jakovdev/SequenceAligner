#pragma once
#ifndef BIO_ALGORITHM_GLOBAL_AFFINE_H
#define BIO_ALGORITHM_GLOBAL_AFFINE_H

#include "bio/types.h"
#include "system/types.h"

void affine_global_init(s32 *restrict match, s32 *restrict gap_x,
			s32 *restrict gap_y, sequence_ptr_t seq1,
			sequence_ptr_t seq2);

void affine_global_fill(s32 *restrict match, s32 *restrict gap_x,
			s32 *restrict gap_y, const s32 *restrict seq1_i,
			sequence_ptr_t seq1, sequence_ptr_t seq2);

#endif /* BIO_ALGORITHM_GLOBAL_AFFINE_H */
