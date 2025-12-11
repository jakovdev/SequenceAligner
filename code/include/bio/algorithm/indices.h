#pragma once
#ifndef BIO_ALGORITHM_INDICES_H
#define BIO_ALGORITHM_INDICES_H

#include "bio/types.h"

extern thread_local s32 *restrict g_seq1_i;

void indices_buffers_init(s32 seq_len_max);

void indices_buffers_free(void);

void indices_precompute(sequence_ptr_t seq);

#endif /* BIO_ALGORITHM_INDICES_H */
