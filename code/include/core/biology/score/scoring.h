#pragma once
#ifndef CORE_BIOLOGY_SCORE_SCORING_H
#define CORE_BIOLOGY_SCORE_SCORING_H

#include <limits.h>

#include "matrices.h"
#include "system/arch.h"

#ifdef USE_SIMD
extern veci_t g_first_row_indices;
extern veci_t g_gap_penalty_vec;
extern veci_t g_gap_open_vec;
extern veci_t g_gap_extend_vec;
#endif

extern int SEQUENCE_LOOKUP[SCHAR_MAX + 1];
extern int SCORING_MATRIX[MAX_MATRIX_DIM][MAX_MATRIX_DIM];

void scoring_matrix_init();

#endif // CORE_BIOLOGY_SCORE_SCORING_H