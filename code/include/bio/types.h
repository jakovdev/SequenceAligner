#pragma once
#ifndef BIO_TYPES_H
#define BIO_TYPES_H

#include <limits.h>
#include <stdint.h>

#include "system/types.h"

#define SEQ_LEN_MAX (INT32_MAX - 1)
#define SEQ_LEN_MIN (1)
#define SEQ_N_MAX (INT32_MAX)
#define SEQ_N_MIN (2)
#define SEQ_LEN_SUM_MIN (SEQ_N_MIN * SEQ_LEN_MIN)
#define SCORE_MIN (INT32_MIN / 2)

#ifndef __cplusplus
extern s32 GAP_PEN;
extern s32 GAP_OPEN;
extern s32 GAP_EXT;

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

enum AlignmentMethod arg_align_method(void);

typedef struct {
	const char *restrict letters;
	s32 length;
} sequence_t;

/* Pointer to a specific sequence */
typedef const sequence_t *const restrict sequence_ptr_t;
#define SEQUENCE_PTR_T(seq) const sequence_t seq[const restrict static 1]
#define SEQ_BAD(seq)                                                     \
	(!seq->letters || !*seq->letters || seq->length < SEQ_LEN_MIN || \
	 seq->length > SEQ_LEN_MAX)
#endif /* __cplusplus */

#endif /* BIO_TYPES_H */
