#pragma once
#ifndef BIO_ALGORITHM_GLOBAL_AFFINE_H
#define BIO_ALGORITHM_GLOBAL_AFFINE_H

#include "bio/types.h"

void affine_global_init(sequence_ptr_t, sequence_ptr_t);

s32 affine_global_fill(sequence_ptr_t, sequence_ptr_t);

#endif /* BIO_ALGORITHM_GLOBAL_AFFINE_H */
