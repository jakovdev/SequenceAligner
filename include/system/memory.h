#ifndef SYSTEM_MEMORY_H
#define SYSTEM_MEMORY_H

#include <stdlib.h>

#include "util/macros.h"

#define KiB ((size_t)1 << 10)
#define MiB (KiB << 10)
#define GiB (MiB << 10)

#define CACHE_LINE ((size_t)64)
#define PAGE_SIZE (4 * KiB)

#ifdef _WIN32
#include <malloc.h>
#define free_aligned _aligned_free
#else
#define free_aligned free
#endif

#define MALLOC(ptr, bytes) ptr = (typeof(ptr))malloc(bytes)
#define MALLOCA(ptr, nmemb) MALLOC(ptr, bytesof(ptr, nmemb))
#define MALLOC_AL(ptr, al, bytes) ptr = (typeof(ptr))alloc_aligned(al, bytes)
#define MALLOCA_AL(ptr, al, nmemb) MALLOC_AL(ptr, al, bytesof(ptr, nmemb))

[[gnu::malloc, gnu::malloc(free_aligned, 1), gnu::alloc_size(2)]]
void *alloc_aligned(size_t alignment, size_t bytes);

size_t available_memory(void);

#endif /* SYSTEM_MEMORY_H */
