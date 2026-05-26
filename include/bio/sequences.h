#ifndef BIO_SEQUENCES_H
#define BIO_SEQUENCES_H

#include <stddef.h>

#include "system/types.h"

struct sequence {
	const char *letters;
	s32 length;
};

struct input {
	struct sequence *seqs;
	s32 *lengths;
	size_t lengths_max;
	s64 *offsets;
	char *letters;
	s64 alignments;
	double average_length;
	s32 seqs_n;
};

[[gnu::nonnull]]
bool input_load(struct input *);
[[gnu::nonnull]]
void input_free(struct input *);
[[gnu::nonnull]]
bool input_lose(struct input *, const bool *lost);

#define SEQ_LEN_MAX (INT32_MAX - 1)
#define SEQ_LEN_MIN (1)
#define SEQ_N_MAX (INT32_MAX)
#define SEQ_N_MIN (2)

#define SEQ_BAD(s)                                                 \
	(!s->letters || !*s->letters || s->length < SEQ_LEN_MIN || \
	 s->length > SEQ_LEN_MAX)

#endif /* BIO_SEQUENCES_H */
