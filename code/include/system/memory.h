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

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

#define MALLOC(ptr, bytes) ptr = malloc(bytes)
#define MALLOCA(ptr, nmemb) MALLOC(ptr, bytesof(ptr, nmemb))
#define MALLOC_CL(ptr, bytes) ptr = alloc_aligned(CACHE_LINE, bytes)
#define MALLOCA_CL(ptr, nmemb) MALLOC_CL(ptr, bytesof(ptr, nmemb))
/* Can continue with else { ... } for failure case */
#define REALLOC(ptr, bytes)                        \
	typeof(ptr) _R##ptr = realloc(ptr, bytes); \
	if likely (_R##ptr)                        \
	ptr = _R##ptr
#define REALLOCA(ptr, nmemb) REALLOC(ptr, bytesof(ptr, nmemb))

/* Free with free_aligned() */
ALLOC void *alloc_aligned(size_t alignment, size_t bytes);

#ifdef _WIN32
#include <malloc.h>
#define free_aligned _aligned_free
#else
#define free_aligned free
#endif

size_t available_memory(void);

#endif /* SYSTEM_MEMORY_H */
