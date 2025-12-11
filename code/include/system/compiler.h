#pragma once
#ifndef SYSTEM_COMPILER_H
#define SYSTEM_COMPILER_H

#ifndef __is_identifier
#define __is_identifier(x) 1
#endif

#define __has_keyword(__x) !(__is_identifier(__x))

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L
#elif defined(_MSC_VER) && _MSC_VER < 1939 && !defined(__clang__)
#error "Unsupported MSVC version, please use a newer one for typeof"
#elif !__has_keyword(typeof)
#define typeof(x) __typeof__(x)
#endif /* C23 has typeof keyword */

#include <stddef.h>
#if !defined(unreachable)
#if defined(_MSC_VER) && !defined(__clang__)
#define unreachable() __assume(0)
#else /* GCC, Clang */
#define unreachable() __builtin_unreachable()
#endif /* MSVC */
#endif /* standard conforming stddef.h macro in C23 */

#ifndef NDEBUG
#include <stdlib.h>
#define unreachable_release() abort()
#else /* Release */
#define unreachable_release() unreachable()
#endif /* NDEBUG */

#if defined(_MSC_VER) && !defined(__clang__)
#define likely(x) (x)
#define unlikely(x) (x)
#ifndef strcasecmp
#define strcasecmp _stricmp
#endif
#else /* GCC, Clang */
#define likely(x) (__builtin_expect(!!(x), 1))
#define unlikely(x) (__builtin_expect(!!(x), 0))
#endif /* MSVC */

#endif /* SYSTEM_COMPILER_H */
