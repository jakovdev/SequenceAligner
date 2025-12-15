#pragma once
#ifndef SYSTEM_COMPILER_H
#define SYSTEM_COMPILER_H

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

#ifdef _WIN32
#if defined(_MSC_VER) && !defined(__clang__)
#define likely(x) (x)
#define unlikely(x) (x)
#define strcasecmp _stricmp
#include <Shlwapi.h>
#elif defined(__MINGW32__) || defined(__MINGW64__) || defined(__clang__)
#define likely(x) (__builtin_expect(!!(x), 1))
#define unlikely(x) (__builtin_expect(!!(x), 0))
#include <shlwapi.h>
#endif
#define strcasestr StrStrIA
#include <stdio.h>
/* https://stackoverflow.com/a/47067149 */
static inline long getdelim(char **buf, size_t *bufsiz, int delim, FILE *fp)
{
	if (!buf || !bufsiz || !fp)
		return -1;

	if (!*buf || !*bufsiz) {
		*bufsiz = BUFSIZ;
		if (!(*buf = malloc(*bufsiz)))
			return -1;
	}

	char *ptr, *eptr;
	for (ptr = *buf, eptr = *buf + *bufsiz;;) {
		int c = fgetc(fp);
		if (c == -1) {
			if (feof(fp)) {
				long diff = (long)(ptr - *buf);
				if (diff) {
					*ptr = '\0';
					return diff;
				}
			}
			return -1;
		}
		*ptr++ = (char)c;
		if (c == delim) {
			*ptr = '\0';
			return (long)(ptr - *buf);
		}
		if (ptr + 2 >= eptr) {
			char *nbuf;
			size_t nbufsiz = *bufsiz * 2;
			long d = (long)(ptr - *buf);
			if (!(nbuf = realloc(*buf, nbufsiz)))
				return -1;
			*buf = nbuf;
			*bufsiz = nbufsiz;
			eptr = nbuf + nbufsiz;
			ptr = nbuf + d;
		}
	}
}

static inline long getline(char **buf, size_t *bufsiz, FILE *fp)
{
	return getdelim(buf, bufsiz, '\n', fp);
}
#else /* GCC, Clang */
#define likely(x) (__builtin_expect(!!(x), 1))
#define unlikely(x) (__builtin_expect(!!(x), 0))
#endif /* MSVC */

#endif /* SYSTEM_COMPILER_H */
