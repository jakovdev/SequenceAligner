#pragma once
#ifndef CORE_BIO_ALGORITHM_GLOBAL_LINEAR_H
#define CORE_BIO_ALGORITHM_GLOBAL_LINEAR_H

#include "core/bio/algorithm/indices.h"
#include "core/bio/types.h"
#include "system/arch.h"

void linear_global_init(score_t* restrict matrix,
                        const sequence_ptr_t seq1,
                        const sequence_ptr_t seq2);

void linear_global_fill(score_t* restrict matrix,
                        const SeqIndices* seq1_indices,
                        const sequence_ptr_t seq1,
                        const sequence_ptr_t seq2);

#ifdef USE_SIMD
void simd_linear_global_row_init(score_t* restrict matrix, const sequence_ptr_t seq1);
#endif

#endif // CORE_BIO_ALGORITHM_GLOBAL_LINEAR_H