#ifndef SYSTEM_TYPES_H
#define SYSTEM_TYPES_H

#include <stdint.h>
#include <stddef.h>

/* alignment sizes, products */
typedef size_t usz;
constexpr usz USZ_MAX = SIZE_MAX;

/* bytes, sequence letters */
typedef uint8_t u8;

#if SIZE_MAX == UINT64_MAX
/* Signed half-width size: scores */
typedef int32_t shz;
constexpr shz SHZ_MAX = INT32_MAX;
constexpr shz SHZ_MIN = INT32_MIN;

/* Unsigned half-width size: sequence lengths, offsets and counts */
typedef uint32_t uhz;
constexpr uhz UHZ_MAX = UINT32_MAX;
#elif SIZE_MAX == UINT32_MAX
/* Signed half-width size: scores */
typedef int16_t shz;
constexpr shz SHZ_MAX = INT16_MAX;
constexpr shz SHZ_MIN = INT16_MIN;

/* Unsigned half-width size: sequence lengths, offsets and counts */
typedef uint16_t uhz;
constexpr uhz UHZ_MAX = UINT16_MAX;
#else
#error "Unknown architecture"
#endif

#ifdef __cplusplus
#define restrict __restrict__
#endif

#ifndef _WIN32
#define SECTION(type, name) aligned(alignof(type)), section(name), used, retain
#else
#define SECTION(type, name) aligned(alignof(type)), section(name), used
#endif

#endif /* SYSTEM_TYPES_H */
