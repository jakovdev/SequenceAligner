#ifndef SYSTEM_TYPES_H
#define SYSTEM_TYPES_H

#include <stdint.h>
#include <stddef.h>

/* scores, sequence lengths, offsets and counts */
typedef int32_t s32;
constexpr s32 S32_MAX = INT32_MAX;
constexpr s32 S32_MIN = INT32_MIN;
/* alignment sizes, products */
typedef int64_t s64;
constexpr s64 S64_MAX = INT64_MAX;
constexpr s64 S64_MIN = INT64_MIN;
/* sequence letters */
typedef uint8_t u8;

#ifdef __cplusplus
#define restrict __restrict__
#endif

#ifndef _WIN32
#define SECTION(type, name) aligned(alignof(type)), section(name), used, retain
#else
#define SECTION(type, name) aligned(alignof(type)), section(name), used
#endif

#endif /* SYSTEM_TYPES_H */
