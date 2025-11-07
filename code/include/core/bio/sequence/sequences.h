#pragma once
#ifndef CORE_BIO_SEQUENCE_SEQUENCES_H
#define CORE_BIO_SEQUENCE_SEQUENCES_H

#include "core/bio/types.h"
#include "system/types.h"

bool sequences_load_from_file(void);

sequence_t *sequences_get(void);

u32 sequences_count(void);

u64 sequences_alignment_count(void);

sequence_t *sequence_get(u32 index);

#ifdef USE_CUDA
char *sequences_flattened(void);

u32 *sequences_offsets(void);

u32 *sequences_lengths(void);

u64 sequences_length_total(void);

u32 sequences_length_max(void);
#endif

#endif // CORE_BIO_SEQUENCE_SEQUENCES_H
