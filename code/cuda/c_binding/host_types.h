#pragma once
#ifndef HOST_TYPES_H
#define HOST_TYPES_H

// CUDA version of types.h
// This file gets used over types.h when compiling with CUDA

#include <stddef.h>
#include <stdint.h>

typedef unsigned int uint;
typedef unsigned long long ull;
typedef signed long long sll;

#define SEQUENCE_LENGTH_MAX (INT32_MAX)
#define SEQUENCE_COUNT_MAX (UINT32_MAX)
#define SEQUENCE_COUNT_MIN (2)
#define SCORE_MIN (INT32_MIN / 2)

/* sequence lengths and counts */
typedef uint32_t u32;
/* scores */
typedef int32_t s32;
/* alignment sizes, counters */
typedef uint64_t u64;
/* checksums */
typedef int64_t s64;
typedef unsigned char uchar;

#endif // HOST_TYPES_H
