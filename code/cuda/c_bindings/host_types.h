#pragma once
#ifndef HOST_TYPES_H
#define HOST_TYPES_H

// CUDA version of biotypes.h
// This file gets used over biotypes.h when compiling with CUDA

#include <stddef.h>
#include <stdint.h>

typedef unsigned long long ull;
typedef signed long long sll;

#if SIZE_MAX == UINT64_MAX
typedef uint32_t HALF_OF_SIZE_T;
typedef uint16_t HALF_OF_HALF_T;
#define HALF_MAX (UINT32_MAX)
#define QUAR_MAX (UINT16_MAX)
#elif SIZE_MAX == UINT32_MAX
typedef uint16_t HALF_OF_SIZE_T;
typedef uint8_t HALF_OF_HALF_T;
#define HALF_MAX (UINT16_MAX)
#define QUAR_MAX (UINT8_MAX)
#else
#error "Unsupported platform: size_t width not 32 or 64 bits"
#endif

typedef HALF_OF_SIZE_T half_t;
typedef HALF_OF_HALF_T quar_t;

// typedef quar_t sequence_length_t;
#define MAX_SEQUENCE_LENGTH (QUAR_MAX)

typedef half_t sequence_index_t;
typedef half_t sequence_count_t;
#define MAX_SEQUENCE_COUNT (HALF_MAX)

typedef size_t alignment_size_t;

typedef half_t sequence_offset_t;

typedef int score_t;

#define SCORE_MIN (INT_MIN / 2)

#endif // HOST_TYPES_H