#include "system/simd.h"

#if defined(__AVX512F__) && defined(__AVX512BW__)
#elif defined(__AVX2__)
#elif defined(__SSE2__)
__m128i _mm_mullo_epi32_fallback(__m128i a, __m128i b)
{
	__m128i tmp1 = _mm_mul_epu32(a, b);
	__m128i tmp2 =
		_mm_mul_epu32(_mm_srli_si128(a, 4), _mm_srli_si128(b, 4));
	return _mm_unpacklo_epi32(
		_mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0, 0, 2, 0)),
		_mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0, 0, 2, 0)));
}

#endif

#if defined(_MSC_VER) && !defined(__clang__)

int __builtin_ctz(unsigned int x)
{
	unsigned long index;
	_BitScanForward(&index, x);
	return (int)index;
}

int __builtin_ctzll(unsigned long long x)
{
	unsigned long index;
	_BitScanForward64(&index, x);
	return (int)index;
}

#endif
