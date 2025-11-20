#pragma once
#ifndef SYSTEM_COMPILER_H
#define SYSTEM_COMPILER_H

#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#define UNREACHABLE() __builtin_unreachable()
#define ALIGN __attribute__((aligned(CACHE_LINE)))
#define ALLOC __attribute__((malloc, alloc_size(1)))
#define UNUSED __attribute__((unused))
#define DESTRUCTOR __attribute__((destructor))
#define PRAGMA(n) _Pragma(#n)
#if defined(__clang__)
#define UNROLL(n) PRAGMA(unroll n)
#define VECTORIZE PRAGMA(clang loop vectorize(assume_safety))
#else /* GCC */
#define UNROLL(n) PRAGMA(GCC unroll n)
#define VECTORIZE PRAGMA(GCC ivdep)
#endif /* clang */
#elif defined(_MSC_VER)
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#define UNREACHABLE() __assume(0)
#define ALIGN __declspec(align(CACHE_LINE))
#define ALLOC
#define UNUSED
#define DESTRUCTOR
#define PRAGMA(n) __pragma(n)
#define UNROLL(n)
#define VECTORIZE PRAGMA(loop(ivdep))
#define strcasecmp _stricmp
#endif

#define OMP_PARALLEL_REDUCTION(var, op)          \
	PRAGMA(omp parallel reduction(op : var)) \
	{
#define OMP_PARALLEL_REDUCTION_END() }

#ifdef _MSC_VER
#define OMP_START_DYNAMIC(var) u32 var = (u32)var##var
#define OMP_FOR_DYNAMIC(var, start, end)  \
	s64 var##var;                     \
	PRAGMA(omp for schedule(dynamic)) \
	for (var##var = (start); var##var < (s64)(end); var##var++)
#else
#define OMP_START_DYNAMIC(var)
#define OMP_FOR_DYNAMIC(var, start, end)  \
	PRAGMA(omp for schedule(dynamic)) \
	for (u32 var = (start); var < (end); var++)
#endif

#endif /* SYSTEM_COMPILER_H */
