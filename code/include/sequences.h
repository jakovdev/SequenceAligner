#pragma once
#ifndef SEQUENCE_H
#define SEQUENCE_H

#include "biotypes.h"

extern void sequences_alloc_from_file(char* start,
                                      char* end,
                                      sequence_count_t total,
                                      float filter,
                                      size_t col);
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

#endif // SEQUENCE_H