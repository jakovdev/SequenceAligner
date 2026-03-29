#pragma once
#ifndef BIO_TYPES_H
#define BIO_TYPES_H

/* NOTE: All EXPANDABLE comments are markers for places which need to be
 *       updated when adding new enum values. 
 */

/* NOTE: Additional alignment methods can be added here if needed.
 *       However, this requires implementing the corresponding algorithm.
 */
enum AlignmentMethod {
	ALIGN_INVALID = -1,
	ALIGN_GOTOH_AFFINE,
	ALIGN_NEEDLEMAN_WUNSCH,
	ALIGN_SMITH_WATERMAN,
	/* NOTE: EXPANDABLE enum AlignmentMethod */
	ALIGN_COUNT
};

#include <limits.h>
#include <stdint.h>

#include "bio/score/matrices.h"
#include "system/types.h"

#define SEQ_LEN_MAX (INT32_MAX - 1)
#define SEQ_LEN_MIN (1)
#define SEQ_N_MAX (INT32_MAX)
#define SEQ_N_MIN (2)
#define SEQ_LEN_SUM_MIN (SEQ_N_MIN * SEQ_LEN_MIN)
#define SCORE_MIN (INT32_MIN / 2)

extern s32 SEQ_LUP[SCHAR_MAX + 1];
extern s32 SUB_MAT[SUBMAT_MAX][SUBMAT_MAX];
extern s32 GAP_PEN;
extern s32 GAP_OPEN;
extern s32 GAP_EXT;

typedef struct {
	const char *restrict letters;
	s32 length;
} sequence_t;

/* Pointer to a specific sequence */
typedef const sequence_t *const restrict sequence_ptr_t;
#define SEQ_INVALID(seq_ptr)                                               \
	(!seq_ptr || !seq_ptr->letters || seq_ptr->length < SEQ_LEN_MIN || \
	 seq_ptr->length > SEQ_LEN_MAX)

enum AlignmentMethod arg_align_method(void);

#endif /* BIO_TYPES_H */
