#pragma once
#ifndef CORE_BIO_SEQUENCE_SEQUENCES_H
#define CORE_BIO_SEQUENCE_SEQUENCES_H

#include "core/bio/types.h"

bool sequences_load_from_file(void);

sequences_t sequences_get(void);

sequence_count_t sequences_count(void);

alignment_size_t sequences_alignment_count(void);

sequence_t *sequence_get(sequence_index_t index);

#ifdef USE_CUDA
char *sequences_flattened(void);

sequence_offset_t *sequences_offsets(void);

quar_t *sequences_lengths(void);

size_t sequences_total_length(void);

sequence_length_t sequences_max_length(void);
#endif

#endif // CORE_BIO_SEQUENCE_SEQUENCES_H
