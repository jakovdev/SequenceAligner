#pragma once
#ifndef CORE_BIO_ALGORITHM_GLOBAL_LINEAR_H
#define CORE_BIO_ALGORITHM_GLOBAL_LINEAR_H

#include "core/bio/types.h"
#include "system/types.h"

void linear_global_init(s32 *restrict matrix, sequence_ptr_t seq1,
			sequence_ptr_t seq2);

void linear_global_fill(s32 *restrict matrix, const s32 *restrict seq1_indices,
			sequence_ptr_t seq1, sequence_ptr_t seq2);

#endif // CORE_BIO_ALGORITHM_GLOBAL_LINEAR_H
