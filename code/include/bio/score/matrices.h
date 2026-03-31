#pragma once
#ifndef BIO_SCORE_MATRICES_H
#define BIO_SCORE_MATRICES_H

#include <limits.h>

#define SEQ_LUT_SIZE (SCHAR_MAX + 1)
#define SUB_MAT_DIM (24)

#ifndef __cplusplus
#include "system/types.h"
#include "system/memory.h"

extern alignas(CACHE_LINE) s32 SEQ_LUT[SEQ_LUT_SIZE];
extern alignas(CACHE_LINE) s32 SUB_MAT[SUB_MAT_DIM][SUB_MAT_DIM];
#endif /* __cplusplus */

#endif /* BIO_SCORE_MATRICES_H */
