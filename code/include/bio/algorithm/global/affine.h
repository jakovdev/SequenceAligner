#pragma once
#ifndef BIO_ALGORITHM_GLOBAL_AFFINE_H
#define BIO_ALGORITHM_GLOBAL_AFFINE_H

#include "bio/types.h"

void affine_global_init(sequence_ptr_t seq1, sequence_ptr_t seq2);

s32 affine_global_fill(sequence_ptr_t seq1, sequence_ptr_t seq2);

#endif /* BIO_ALGORITHM_GLOBAL_AFFINE_H */
