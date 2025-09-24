#pragma once
#ifndef CORE_BIOLOGY_ALGORITHM_ALIGNMENT_H
#define CORE_BIOLOGY_ALGORITHM_ALIGNMENT_H

#include "core/biology/types.h"

score_t align_pairwise(const sequence_ptr_t seq1, const sequence_ptr_t seq2);

#endif // CORE_BIOLOGY_ALGORITHM_ALIGNMENT_H