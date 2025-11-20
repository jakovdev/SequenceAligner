#pragma once
#ifndef BIO_TYPES_H
#define BIO_TYPES_H

/* NOTE: All EXPANDABLE comments are markers for places which need to be
 *       updated when adding new enum values. 
 */

/* NOTE: Additional sequence types can be added here if needed.
 *       However, this requires defining the corresponding alphabet
 *       and substitution matrices.
 */
enum SequenceType {
	SEQ_TYPE_INVALID = -1,
	SEQ_TYPE_AMINO,
	SEQ_TYPE_NUCLEO,
	/* NOTE: EXPANDABLE enum SequenceType */
	SEQ_TYPE_COUNT
};

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

/* NOTE: Additional types can be added here if needed.
 *       However, this requires implementing new arguments.
 */
enum GapPenaltyType {
	GAP_TYPE_LINEAR,
	GAP_TYPE_AFFINE,
	/* NOTE: EXPANDABLE enum GapPenaltyType */
};

#include <stdbool.h>
#include <stdint.h>

#include "system/types.h"

#define SEQUENCE_LENGTH_MAX (INT32_MAX)
#define SEQUENCE_COUNT_MAX (UINT32_MAX)
#define SEQUENCE_COUNT_MIN (2)
#define SCORE_MIN (INT32_MIN / 2)

typedef struct {
	char *letters;
	u64 length;
} sequence_t;

/* Pointer to a specific sequence */
typedef const sequence_t *const restrict sequence_ptr_t;

typedef s32 (*align_func_t)(sequence_ptr_t, sequence_ptr_t);
align_func_t align_function(enum AlignmentMethod method);

enum SequenceType arg_sequence_type(void);
int arg_sub_matrix(void);

enum AlignmentMethod arg_align_method(void);
s32 arg_gap_pen(void);
s32 arg_gap_open(void);
s32 arg_gap_ext(void);

#endif /* BIO_TYPES_H */
