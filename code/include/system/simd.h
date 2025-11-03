#pragma once
#ifndef SYSTEM_SIMD_H
#define SYSTEM_SIMD_H

#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>
#ifndef _MSC_VER
#include <x86intrin.h>
#endif
#define USE_SIMD
#define USE_AVX512
typedef __m512i veci_t;
typedef __mmask64 num_t;
#define BYTES (64)
#define NUM_ELEMS (16)
#define ctz(x) ((num_t)__builtin_ctzll(x))
#define loadu _mm512_loadu_si512
#define storeu _mm512_storeu_si512
#define add_epi32 _mm512_add_epi32
#define sub_epi32 _mm512_sub_epi32
#define mullo_epi32 _mm512_mullo_epi32
#define set1_epi32 _mm512_set1_epi32
#define set1_epi8 _mm512_set1_epi8
#define cmpeq_epi8 _mm512_cmpeq_epi8_mask
#define movemask_epi8
#define or_mask(a, b) ((a) | (b))
#define or_si _mm512_or_si512
#define setzero_si _mm512_setzero_si512
#define and_si _mm512_and_si512
#define setr_epi32 _mm512_setr_epi32
#define set_row_indices() \
	_mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)

#elif defined(__AVX2__)
#include <immintrin.h>
#include <stdint.h>
#define USE_SIMD
#define USE_AVX2
typedef __m256i veci_t;
typedef uint32_t num_t;
#define BYTES (32)
#define NUM_ELEMS (8)
#define ctz(x) ((num_t)__builtin_ctz(x))
#define loadu _mm256_loadu_si256
#define storeu _mm256_storeu_si256
#define add_epi32 _mm256_add_epi32
#define sub_epi32 _mm256_sub_epi32
#define mullo_epi32 _mm256_mullo_epi32
#define set1_epi32 _mm256_set1_epi32
#define set1_epi8 _mm256_set1_epi8
#define cmpeq_epi8 _mm256_cmpeq_epi8
#define movemask_epi8(x) ((num_t)_mm256_movemask_epi8(x))
#define or_si _mm256_or_si256
#define setzero_si _mm256_setzero_si256
#define and_si _mm256_and_si256
#define setr_epi32 _mm256_setr_epi32
#define set_row_indices() _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8)

#elif defined(__SSE2__)
#include <emmintrin.h>
#include <stdint.h>
#define USE_SIMD
#define USE_SSE
typedef __m128i veci_t;
typedef uint16_t num_t;
#define BYTES (16)
#define NUM_ELEMS (4)
#define ctz(x) ((num_t)__builtin_ctz(x))
#define loadu _mm_loadu_si128
#define storeu _mm_storeu_si128
#define add_epi32 _mm_add_epi32
#define sub_epi32 _mm_sub_epi32
#define mullo_epi32 _mm_mullo_epi32_fallback
#define set1_epi32 _mm_set1_epi32
#define set1_epi8 _mm_set1_epi8
#define cmpeq_epi8 _mm_cmpeq_epi8
#define movemask_epi8(x) ((num_t)_mm_movemask_epi8(x))
#define or_si _mm_or_si128
#define setzero_si _mm_setzero_si128
#define and_si _mm_and_si128
#define setr_epi32 _mm_setr_epi32
#define set_row_indices() _mm_setr_epi32(1, 2, 3, 4)

__m128i _mm_mullo_epi32_fallback(__m128i a, __m128i b);

#else
#define prefetch(x)
#define prefetch_write(x)
#endif

#ifdef USE_SIMD
#define PREFETCH_DISTANCE (BYTES << 4)
#define prefetch(x) _mm_prefetch((const char *)(x), _MM_HINT_T0)
#define prefetch_write(x) _mm_prefetch((const char *)(x), _MM_HINT_T1)
#endif

#ifdef _MSC_VER
#include <intrin.h>
#define __builtin_popcount(x) ((int)__popcnt(x))
#define __builtin_popcountll(x) ((int)__popcnt64(x))
int __builtin_ctz(unsigned int x);
int __builtin_ctzll(unsigned long long x);
#endif

#endif // SYSTEM_SIMD_H
