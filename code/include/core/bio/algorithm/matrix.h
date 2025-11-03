#pragma once
#ifndef CORE_BIO_ALGORITHM_MATRIX_H
#define CORE_BIO_ALGORITHM_MATRIX_H

#include "core/bio/types.h"
#include "system/arch.h"

#define MAX_STACK_SEQUENCE_LENGTH (4 * KiB)

#define STACK_MATRIX_THRESHOLD (128 * KiB)
#define MATRIX_SIZE(len1, len2) ((len1 + 1U) * (len2 + 1U))
#define MATRICES_3X_SIZE(len1, len2) (3 * MATRIX_SIZE(len1, len2))
#define MATRIX_BYTES(len1, len2) (MATRIX_SIZE(len1, len2) * sizeof(score_t))
#define MATRICES_3X_BYTES(len1, len2) (3 * MATRIX_BYTES(len1, len2))
#define USE_STACK_MATRIX(bytes) ((bytes) <= STACK_MATRIX_THRESHOLD)

score_t *matrix_alloc(score_t *stack_matrix, size_t bytes);

void matrix_free(score_t *matrix, score_t *stack_matrix);

#endif // CORE_BIO_ALGORITHM_MATRIX_H
