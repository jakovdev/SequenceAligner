#pragma once
#ifndef SYSTEM_MEMORY_H
#define SYSTEM_MEMORY_H

#include <stdlib.h>

#include "system/compiler.h"

#define KiB ((size_t)1 << 10)
#define MiB (KiB << 10)
#define GiB (MiB << 10)

#define CACHE_LINE ((size_t)64)
#define PAGE_SIZE (4 * KiB)

#define sizeof_field(t, f) (sizeof(((t *)0)->f))
#define bytesof(ptr, nmemb) (sizeof(*(ptr)) * (nmemb))

#ifdef _WIN32
#include <malloc.h>
#define free_aligned _aligned_free
#define realloc_aligned(p, alignment, size) _aligned_realloc(p, size, alignment)
#define _realloc_aligned_free(oldp, newp, old, new) \
	do {                                        \
		(void)(oldp);                       \
		(void)(newp);                       \
		(void)(old);                        \
		(void)(new);                        \
	} while (0)
#else
#include <string.h>
#define free_aligned free
#define realloc_aligned(p, alignment, size) alloc_aligned(alignment, size)
#define _realloc_aligned_free(oldp, newp, old, new)        \
	do {                                               \
		memcpy(newp, oldp, old < new ? old : new); \
		free_aligned(oldp);                        \
	} while (0)
#endif

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#define _MCAT_2(a, b) a##b
#define _MCAT(a, b) _MCAT_2(a, b)

#define MALLOC(ptr, bytes) ptr = malloc(bytes)
#define MALLOCA(ptr, nmemb) MALLOC(ptr, bytesof(ptr, nmemb))
#define MALLOC_AL(ptr, al, bytes) ptr = alloc_aligned(al, bytes)
#define MALLOCA_AL(ptr, al, nmemb) MALLOC_AL(ptr, al, bytesof(ptr, nmemb))
/* Continue with { ... } for failure case */
#define REALLOC(ptr, bytes)                               \
	void *_MCAT(_NM, __LINE__) = realloc(ptr, bytes); \
	if likely (_MCAT(_NM, __LINE__)) {                \
		ptr = _MCAT(_NM, __LINE__);               \
	} else
#define REALLOCA(ptr, nmemb) REALLOC(ptr, bytesof(ptr, nmemb))
#define REALLOC_AL(ptr, al, oldb, newb)                                       \
	void *_MCAT(_NM, __LINE__) = realloc_aligned(ptr, al, newb);          \
	if likely (_MCAT(_NM, __LINE__)) {                                    \
		_realloc_aligned_free(ptr, _MCAT(_NM, __LINE__), oldb, newb); \
		ptr = _MCAT(_NM, __LINE__);                                   \
	} else
#define REALLOCA_AL(ptr, al, oldn, newn) \
	REALLOC_AL(ptr, al, bytesof(ptr, oldn), bytesof(ptr, newn))

/* Free with free_aligned() */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((malloc, alloc_size(2)))
#endif
void *alloc_aligned(size_t alignment, size_t bytes);

size_t available_memory(void);

#endif /* SYSTEM_MEMORY_H */
