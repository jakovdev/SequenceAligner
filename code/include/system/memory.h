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
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#define ALLOC_BYTES(ptr, n) (sizeof(*(ptr)) * (n))

#define MALLOC(ptr, n) ptr = malloc(ALLOC_BYTES(ptr, n))
#define MALLOC_CL(ptr, n) ptr = alloc_aligned(CACHE_LINE, ALLOC_BYTES(ptr, n))
/* Can continue with else { ... } for failure case */
#define REALLOC(ptr, n)                                          \
	TYPEOF(ptr) _R##ptr = realloc(ptr, ALLOC_BYTES(ptr, n)); \
	if (_R##ptr)                                             \
	ptr = _R##ptr

/* Free with free_aligned() */
ALLOC void *alloc_aligned(size_t alignment, size_t bytes);

size_t available_memory(void);

#endif /* SYSTEM_MEMORY_H */
