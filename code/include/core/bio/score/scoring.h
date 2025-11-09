#pragma once
#ifndef CORE_BIO_SCORE_SCORING_H
#define CORE_BIO_SCORE_SCORING_H

#include <limits.h>

#include "matrices.h"
// #include "system/simd.h"

#if 0 // USE_SIMD == 1
extern veci_t g_first_row_indices;
extern veci_t g_gap_pen_vec;
extern veci_t g_gap_open_vec;
extern veci_t g_gap_ext_vec;
#endif

extern int SEQ_LUP[SCHAR_MAX + 1];
extern int SUB_MAT[SUBMAT_MAX][SUBMAT_MAX];

void scoring_init(void);

#endif // CORE_BIO_SCORE_SCORING_H
