#pragma once
#ifndef HOST_TYPES_H
#define HOST_TYPES_H

/*
 * Increase if needed, this depends on your available VRAM.
 * On an 8GB card, the limit is around 20000-22000.
 * If over the limit, it will write out of memory error message.
 */
#define MAX_CUDA_SEQUENCE_LENGTH (1024)

#define SUB_MATDIM (24)
#define SEQ_LUPSIZ (128)

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

typedef unsigned int uint;
typedef unsigned long long ull;
typedef signed long long sll;

/* sequence lengths and counts */
typedef uint32_t u32;
/* scores */
typedef int32_t s32;
/* alignment sizes, counters */
typedef uint64_t u64;
/* checksums */
typedef int64_t s64;
typedef unsigned char uchar;

#ifndef CORE_BIO_TYPES_H
#define SEQUENCE_LENGTH_MAX (INT32_MAX)
#define SEQUENCE_COUNT_MAX (UINT32_MAX)
#define SEQUENCE_COUNT_MIN (2)
#define SCORE_MIN (INT32_MIN / 2)

typedef struct {
	char *letters;
	u64 length;
} sequence_t;
#endif // CORE_BIO_TYPES_H

#endif // HOST_TYPES_H
