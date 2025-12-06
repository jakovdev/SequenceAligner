#pragma once
#ifndef HOST_TYPES_H
#define HOST_TYPES_H

/*
 * Increase if needed, this depends on your available VRAM.
 * On an 8GB card, the limit is around 20000-22000.
 * If over the limit, it will write out of memory error message.
 */
#define MAX_CUDA_SEQUENCE_LENGTH (1023)

#define SUB_MATDIM (24)
#define SUB_MATSIZE (SUB_MATDIM * SUB_MATDIM)
#define SEQ_LUPSIZ (128)

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

/* scores, sequence lengths and counts */
typedef int32_t s32;
/* checksums, alignment sizes, counters, products */
typedef int64_t s64;

typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned long long ull;
typedef signed long long sll;

#ifndef BIO_TYPES_H
#define SEQUENCE_LENGTH_MAX (INT32_MAX - 1)
#define SEQUENCE_COUNT_MAX (INT32_MAX)
#define SEQUENCE_COUNT_MIN (2)
#define SCORE_MIN (INT32_MIN / 2)

typedef struct {
	char *letters;
	s32 length;
} sequence_t;
#endif /* BIO_TYPES_H */

#endif /* HOST_TYPES_H */
