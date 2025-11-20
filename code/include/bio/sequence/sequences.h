#pragma once
#ifndef BIO_SEQUENCE_SEQUENCES_H
#define BIO_SEQUENCE_SEQUENCES_H

#include "bio/types.h"
#include "system/types.h"

bool sequences_load_from_file(void);

sequence_t *sequences(void);

u32 sequences_count(void);

u64 sequences_alignment_count(void);

sequence_t *sequence(u32 index);

#ifdef USE_CUDA
u64 sequences_length_sum(void);
u32 sequences_length_max(void);
#endif

#endif /* BIO_SEQUENCE_SEQUENCES_H */
