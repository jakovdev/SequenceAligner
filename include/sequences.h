#ifndef SEQUENCE_H
#define SEQUENCE_H

#include "stdbool.h"
#include "stddef.h"

typedef struct
{
    char* letters;
    size_t length;
} sequence_t;

extern void sequences_alloc_from_file(char* file_cursor,
                                      char* file_end,
                                      size_t sequences_total,
                                      float filter_threshold,
                                      bool apply_filtering,
                                      int sequence_column);

extern void sequences_get_pair(size_t first_index,
                               char* restrict* first_sequence_out,
                               size_t* restrict first_length_out,
                               size_t second_index,
                               char* restrict* second_sequence_out,
                               size_t* restrict second_length_out);

extern sequence_t* sequences_get(void);
extern size_t sequences_count(void);
extern size_t sequences_alignment_count(void);

#endif // SEQUENCE_H