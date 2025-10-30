#pragma once
#ifndef CORE_BIO_ALGORITHM_LOCAL_AFFINE_H
#define CORE_BIO_ALGORITHM_LOCAL_AFFINE_H

#include "core/bio/algorithm/indices.h"
#include "core/bio/types.h"
#include "system/simd.h"

void affine_local_init(score_t* restrict match,
                       score_t* restrict gap_x,
                       score_t* restrict gap_y,
                       sequence_ptr_t seq1,
                       sequence_ptr_t seq2);

score_t affine_local_fill(score_t* restrict match,
                          score_t* restrict gap_x,
                          score_t* restrict gap_y,
                          const SeqIndices* seq1_indices,
                          sequence_ptr_t seq1,
                          sequence_ptr_t seq2);

#ifdef USE_SIMD
void simd_affine_local_row_init(score_t* restrict match,
                                score_t* restrict gap_x,
                                score_t* restrict gap_y,
                                sequence_ptr_t seq1,
                                sequence_ptr_t seq2);
#endif

#endif // CORE_BIO_ALGORITHM_LOCAL_AFFINE_H
