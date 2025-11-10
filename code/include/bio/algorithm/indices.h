#pragma once
#ifndef BIO_ALGORITHM_INDICES_H
#define BIO_ALGORITHM_INDICES_H

#include <stdbool.h>

#include "bio/types.h"

void seq_indices_precompute(s32 *restrict indices, sequence_ptr_t seq);

void seq_indices_free(s32 *restrict indices, bool is_stack);

#endif // BIO_ALGORITHM_INDICES_H
