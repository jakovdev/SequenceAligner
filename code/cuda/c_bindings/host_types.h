#pragma once
#ifndef HOST_TYPES_H
#define HOST_TYPES_H

#include <stdint.h>

typedef unsigned long long ull;
typedef signed long long sll;

#if SIZE_MAX == UINT64_MAX
typedef uint32_t HALF_OF_SIZE_T;
#elif SIZE_MAX == UINT32_MAX
typedef uint16_t HALF_OF_SIZE_T;
#elif SIZE_MAX == UINT16_MAX
typedef uint8_t HALF_OF_SIZE_T;
#else
#error "Unsupported platform: size_t width not 16/32/64 bits"
#endif

typedef HALF_OF_SIZE_T half_t;

#endif // HOST_TYPES_H