#pragma once
#ifndef BIO_SEQUENCE_SEQUENCES_H
#define BIO_SEQUENCE_SEQUENCES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

#include "system/types.h"

struct sequence {
	const char *restrict letters;
	s32 length;
};

extern s64 ALIGNMENTS;
extern s32 *LENGTHS;
extern s64 *OFFSETS;
extern char *LETTERS;
extern struct sequence *SEQS;
extern s32 SEQS_N;

bool sequences_load_from_file(void);

#define SEQ_LEN_MAX (INT32_MAX - 1)
#define SEQ_LEN_MIN (1)
#define SEQ_N_MAX (INT32_MAX)
#define SEQ_N_MIN (2)
#define SCORE_MIN (INT32_MIN / 2)
#define SEQ_LEN_SUM_MIN (SEQ_N_MIN * SEQ_LEN_MIN)

extern size_t LENGTHS_MAX;
#define TABLE_SIZE ((LENGTHS_MAX + 1) * (LENGTHS_MAX + 1))

typedef const struct sequence *const restrict seq_ptr;
#define SEQ_PTR(s) const struct sequence s[const restrict static 1]
#define SEQ_BAD(s)                                                 \
	(!s->letters || !*s->letters || s->length < SEQ_LEN_MIN || \
	 s->length > SEQ_LEN_MAX)

#ifdef __cplusplus
}
#endif

#endif /* BIO_SEQUENCE_SEQUENCES_H */
