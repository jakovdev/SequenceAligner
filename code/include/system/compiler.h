#pragma once
#ifndef SYSTEM_COMPILER_H
#define SYSTEM_COMPILER_H

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L
#elif defined(_MSC_VER) && _MSC_VER < 1939 && !defined(__clang__)
#define typeof decltype
#elif !defined(typeof)
#define typeof __typeof__
#endif /* C23 has typeof keyword */

#if !defined(unreachable)
#ifdef NDEBUG
#if defined(_MSC_VER) && !defined(__clang__)
#define unreachable() __assume(0)
#else /* GCC, Clang */
#define unreachable() __builtin_unreachable()
#endif /* MSVC */
#else /* Debug */
#include <stdlib.h>
#define unreachable() abort()
#endif /* NDEBUG */
#endif /* standard conforming stddef.h macro in C23 */

#if defined(_MSC_VER) && !defined(__clang__)
#define likely(x) (x)
#define unlikely(x) (x)
#define ALIGN(x) __declspec(align(x))
#define ALLOC
#define PRAGMA(n) __pragma(n)
#define UNROLL(n)
#define VECTORIZE PRAGMA(loop(ivdep))
#define strcasecmp _stricmp
#else /* GCC, Clang */
#define likely(x) (__builtin_expect(!!(x), 1))
#define unlikely(x) (__builtin_expect(!!(x), 0))
#define ALIGN(x) __attribute__((aligned(x)))
#define ALLOC __attribute__((malloc, alloc_size(1)))
#define PRAGMA(n) _Pragma(#n)
#if defined(__clang__)
#define UNROLL(n) PRAGMA(unroll n)
#define VECTORIZE PRAGMA(clang loop vectorize(assume_safety))
#elif defined(__GNUC__)
#define UNROLL(n) PRAGMA(GCC unroll n)
#define VECTORIZE PRAGMA(GCC ivdep)
#endif /* Clang */
#endif /* MSVC */

#endif /* SYSTEM_COMPILER_H */
