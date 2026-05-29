#ifndef BIO_SEQUENCE_H
#define BIO_SEQUENCE_H

#include "system/types.h"

struct sequence {
	const char *letters;
	s32 length;
};

constexpr s32 SEQ_LEN_MAX = (S32_MAX - 1);
constexpr s32 SEQ_LEN_MIN = 1;
constexpr s32 SEQ_N_MAX = S32_MAX;
constexpr s32 SEQ_N_MIN = 2;

#define SEQ_BAD(s)                                                 \
	(!s->letters || !*s->letters || s->length < SEQ_LEN_MIN || \
	 s->length > SEQ_LEN_MAX)

#endif /* BIO_SEQUENCE_H */
