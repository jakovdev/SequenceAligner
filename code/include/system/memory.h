#pragma once
#ifndef SYSTEM_MEMORY_H
#define SYSTEM_MEMORY_H

#include <stdlib.h>

#include "system/compiler.h"

#define KiB ((size_t)1 << 10)
#define MiB (KiB << 10)
#define GiB (MiB << 10)

#define CACHE_LINE ((size_t)64)

#define sizeof_field(t, f) (sizeof(((t *)0)->f))
#define bytesof(ptr, nmemb) (sizeof(*(ptr)) * (nmemb))

#ifdef _WIN32
#include <malloc.h>
#define free_aligned _aligned_free
#define realloc_aligned(p, alignment, size) _aligned_realloc(p, size, alignment)
#define _realloc_aligned_free(oldp, newp, old, new)
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

#define MALLOC(ptr, bytes) ptr = malloc(bytes)
#define MALLOCA(ptr, nmemb) MALLOC(ptr, bytesof(ptr, nmemb))
#define MALLOC_CL(ptr, bytes) ptr = alloc_aligned(CACHE_LINE, bytes)
#define MALLOCA_CL(ptr, nmemb) MALLOC_CL(ptr, bytesof(ptr, nmemb))
/* Continue with { ... } for failure case */
#define REALLOC(ptr, bytes)                \
	void *_Rmem = realloc(ptr, bytes); \
	if likely (_Rmem) {                \
		ptr = _Rmem;               \
	} else
#define REALLOCA(ptr, nmemb) REALLOC(ptr, bytesof(ptr, nmemb))
#define REALLOC_CL(ptr, oldb, newb)                            \
	void *_Rmem = realloc_aligned(ptr, CACHE_LINE, newb);  \
	if likely (_Rmem) {                                    \
		_realloc_aligned_free(ptr, _Rmem, oldb, newb); \
		ptr = _Rmem;                                   \
	} else
#define REALLOCA_CL(ptr, oldn, newn) \
	REALLOC_CL(ptr, bytesof(ptr, oldn), bytesof(ptr, newn))

/* Free with free_aligned() */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((malloc, alloc_size(2)))
#endif
void *alloc_aligned(size_t alignment, size_t bytes);

size_t available_memory(void);

#endif /* SYSTEM_MEMORY_H */
