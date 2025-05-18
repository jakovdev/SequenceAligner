#pragma once
#ifndef SEQUENCE_H
#define SEQUENCE_H

#include "stdbool.h"
#include "stddef.h"

typedef struct
{
    char* letters;
    size_t length;
} sequence_t;

extern void sequences_alloc_from_file(char* start, char* end, size_t total, float filter, int col);
extern sequence_t* sequences_get(void);
extern size_t sequences_count(void);
extern size_t sequences_alignment_count(void);

extern void sequences_get_pair(size_t first_index,
                               char* restrict* first_sequence_out,
                               size_t* restrict first_length_out,
                               size_t second_index,
                               char* restrict* second_sequence_out,
                               size_t* restrict second_length_out);

#ifdef USE_CUDA
#include "host_types.h"
extern char* sequences_flattened(void);
extern half_t* sequences_offsets(void);
extern half_t* sequences_lengths(void);
extern size_t sequences_total_length(void);
extern size_t sequences_max_length(void);
#endif

#endif // SEQUENCE_H