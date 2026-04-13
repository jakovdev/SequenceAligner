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

extern enum AlignmentMethod METHOD;

#ifndef __cplusplus
extern s32 GAP_PEN;
extern s32 GAP_OPEN;
extern s32 GAP_EXT;

struct seq {
	const char *restrict letters;
	s32 length;
};

/* Pointer to a specific sequence */
typedef const struct seq *const restrict seq_ptr;
#define SEQ_PTR(s) const struct seq s[const restrict static 1]
#define SEQ_BAD(s)                                                 \
	(!s->letters || !*s->letters || s->length < SEQ_LEN_MIN || \
	 s->length > SEQ_LEN_MAX)
#endif /* __cplusplus */

#endif /* BIO_TYPES_H */
