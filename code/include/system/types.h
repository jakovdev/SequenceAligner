#pragma once
#ifndef SYSTEM_TYPES_H
#define SYSTEM_TYPES_H

#include <stdint.h>
#include <inttypes.h>

/* scores, sequence lengths and counts */
typedef int32_t s32;
#define Ps32 "%" PRId32
/* checksums, alignment sizes, counters, products */
typedef int64_t s64;
#define Ps64 "%" PRId64
/* small range stuff */
typedef uint8_t u8;
#define Pu8 "%" PRIu8

typedef unsigned char uchar;
typedef unsigned long long ull;

#endif /* SYSTEM_TYPES_H */
