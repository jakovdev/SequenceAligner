#pragma once
#ifndef CORE_BIOLOGY_SEQUENCE_SEQUENCES_H
#define CORE_BIOLOGY_SEQUENCE_SEQUENCES_H

#include "core/biology/types.h"
#include "core/io/files.h"

extern bool sequences_alloc_from_file(FileTextPtr input_file);
extern sequences_t sequences_get(void);
extern sequence_count_t sequences_count(void);
extern alignment_size_t sequences_alignment_count(void);

extern void sequences_get_pair(sequence_index_t first_sequence_index,
                               sequence_ptr_t* restrict first_sequence_out,
                               sequence_index_t second_index,
                               sequence_ptr_t* restrict second_sequence_out);

#ifdef USE_CUDA
extern char* sequences_flattened(void);
extern sequence_offset_t* sequences_offsets(void);
extern quar_t* sequences_lengths(void);
extern size_t sequences_total_length(void);
extern sequence_length_t sequences_max_length(void);
#endif

#endif // CORE_BIOLOGY_SEQUENCE_SEQUENCES_H