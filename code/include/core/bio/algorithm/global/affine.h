#pragma once
#ifndef CORE_BIO_ALGORITHM_GLOBAL_AFFINE_H
#define CORE_BIO_ALGORITHM_GLOBAL_AFFINE_H

#include "core/bio/algorithm/indices.h"
#include "core/bio/types.h"

void affine_global_init(score_t* restrict match,
                        score_t* restrict gap_x,
                        score_t* restrict gap_y,
                        const sequence_ptr_t seq1,
                        const sequence_ptr_t seq2);

void affine_global_fill(score_t* restrict match,
                        score_t* restrict gap_x,
                        score_t* restrict gap_y,
                        const SeqIndices* seq1_indices,
                        const sequence_ptr_t seq1,
                        const sequence_ptr_t seq2);

#endif // CORE_BIO_ALGORITHM_GLOBAL_AFFINE_H
