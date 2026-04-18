#ifndef BIO_SCORE_MATRICES_H
#define BIO_SCORE_MATRICES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <limits.h>

#define SEQ_LUT_SIZE (SCHAR_MAX + 1)
#define SUB_MAT_DIM (24)

#include "system/types.h"

extern s32 SEQ_LUT[SEQ_LUT_SIZE];
extern s32 SUB_MAT[SUB_MAT_DIM][SUB_MAT_DIM];

#ifdef __cplusplus
}
#endif

#endif /* BIO_SCORE_MATRICES_H */
