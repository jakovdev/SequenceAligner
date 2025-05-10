#pragma once
#ifndef SCORING_H
#define SCORING_H

#include "arch.h"
#include "matrices.h"

#include <limits.h>

#ifdef USE_SIMD
extern veci_t g_first_row_indices;
extern veci_t g_gap_penalty_vec;
extern veci_t g_gap_start_vec;
extern veci_t g_gap_extend_vec;
#endif

extern int SEQUENCE_LOOKUP[SCHAR_MAX + 1];
extern int SCORING_MATRIX[MAX_MATRIX_DIM][MAX_MATRIX_DIM];

void scoring_matrix_init();

#endif