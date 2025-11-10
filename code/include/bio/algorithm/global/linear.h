#pragma once
#ifndef BIO_ALGORITHM_GLOBAL_LINEAR_H
#define BIO_ALGORITHM_GLOBAL_LINEAR_H

#include "bio/types.h"
#include "system/types.h"

void linear_global_init(s32 *restrict matrix, sequence_ptr_t seq1,
			sequence_ptr_t seq2);

void linear_global_fill(s32 *restrict matrix, const s32 *restrict seq1_i,
			sequence_ptr_t seq1, sequence_ptr_t seq2);

#endif // BIO_ALGORITHM_GLOBAL_LINEAR_H
