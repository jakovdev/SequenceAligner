#pragma once
#ifndef CORE_BIO_ALGORITHM_INDICES_H
#define CORE_BIO_ALGORITHM_INDICES_H

#include <stdbool.h>

#include "core/bio/types.h"

void seq_indices_precompute(s32 *restrict indices, sequence_ptr_t seq);

void seq_indices_free(s32 *restrict indices, bool is_stack);

#endif // CORE_BIO_ALGORITHM_INDICES_H
