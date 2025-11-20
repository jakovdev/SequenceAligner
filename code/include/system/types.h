#pragma once
#ifndef SYSTEM_TYPES_H
#define SYSTEM_TYPES_H

#include <stdint.h>
#include <inttypes.h>

/* sequence counts and lenghts (CUDA) */
typedef uint32_t u32;
#define Pu32 "%" PRIu32
/* scores */
typedef int32_t s32;
#define Ps32 "%" PRId32
/* alignment sizes, counters, sequence lengths (CPU) */
typedef uint64_t u64;
#define Pu64 "%" PRIu64
/* checksums */
typedef int64_t s64;
#define Ps64 "%" PRId64
/* small range stuff */
typedef uint8_t u8;
#define Pu8 "%" PRIu8

typedef unsigned char uchar;
typedef unsigned long ul;
typedef unsigned long long ull;

#endif /* SYSTEM_TYPES_H */
