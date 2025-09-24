#pragma once
#ifndef CORE_BIOLOGY_ALGORITHM_INDICES_H
#define CORE_BIOLOGY_ALGORITHM_INDICES_H

#include <stdbool.h>

#include "core/biology/types.h"

typedef struct
{
    int* data;
    bool is_stack;
} SeqIndices;

void seq_indices_precompute(SeqIndices* indices, const sequence_ptr_t seq);
void seq_indices_free(SeqIndices* indices);

#endif // CORE_BIOLOGY_ALGORITHM_INDICES_H