#pragma once
#ifndef BIO_ALGORITHM_GLOBAL_LINEAR_H
#define BIO_ALGORITHM_GLOBAL_LINEAR_H

#include "bio/types.h"
#include "system/types.h"

void linear_global_init(sequence_ptr_t seq1, sequence_ptr_t seq2);

s32 linear_global_fill(sequence_ptr_t seq1, sequence_ptr_t seq2);

#endif /* BIO_ALGORITHM_GLOBAL_LINEAR_H */
