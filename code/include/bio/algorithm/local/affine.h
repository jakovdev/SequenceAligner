#pragma once
#ifndef BIO_ALGORITHM_LOCAL_AFFINE_H
#define BIO_ALGORITHM_LOCAL_AFFINE_H

#include "bio/types.h"

void affine_local_init(sequence_ptr_t, sequence_ptr_t);

s32 affine_local_fill(sequence_ptr_t, sequence_ptr_t);

#endif /* BIO_ALGORITHM_LOCAL_AFFINE_H */
