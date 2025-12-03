#pragma once
#ifndef BIO_SEQUENCE_SEQUENCES_H
#define BIO_SEQUENCE_SEQUENCES_H

#include "bio/types.h"

bool sequences_load_from_file(void);

sequence_t *sequences(void);

u32 sequences_count(void);

u64 sequences_alignment_count(void);

sequence_t *sequence(u32 index);

u32 sequences_length_max(void);

#ifdef USE_CUDA
u64 sequences_length_sum(void);
#endif

#endif /* BIO_SEQUENCE_SEQUENCES_H */
