#pragma once
#ifndef BIO_ALGORITHM_INDICES_H
#define BIO_ALGORITHM_INDICES_H

#include "bio/types.h"
#include "system/types.h"

extern _Thread_local s32 *restrict g_seq1_i;

void indices_buffers_init(u32 lenmax);

void indices_buffers_free(void);

void indices_precompute(sequence_ptr_t seq);

#endif /* BIO_ALGORITHM_INDICES_H */
