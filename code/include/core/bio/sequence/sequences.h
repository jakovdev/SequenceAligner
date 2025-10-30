#pragma once
#ifndef CORE_BIO_SEQUENCE_SEQUENCES_H
#define CORE_BIO_SEQUENCE_SEQUENCES_H

#include "core/bio/types.h"

extern bool sequences_load_from_file(void);
extern sequences_t sequences_get(void);
extern sequence_count_t sequences_count(void);
extern alignment_size_t sequences_alignment_count(void);
extern sequence_t* sequence_get(sequence_index_t index);

#ifdef USE_CUDA
extern char* sequences_flattened(void);
extern sequence_offset_t* sequences_offsets(void);
extern quar_t* sequences_lengths(void);
extern size_t sequences_total_length(void);
extern sequence_length_t sequences_max_length(void);
#endif

#endif // CORE_BIO_SEQUENCE_SEQUENCES_H
