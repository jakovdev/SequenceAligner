#pragma once
#ifndef SYSTEM_MEMORY_H
#define SYSTEM_MEMORY_H

#ifdef _WIN32
#include <malloc.h>
#else
#include <alloca.h>
#endif

#include <stdlib.h>

#include "system/compiler.h"

#define KiB ((size_t)1 << 10)
#define MiB (KiB << 10)
#define GiB (MiB << 10)

#define CACHE_LINE ((size_t)64)

#define sizeof_field(t, f) (sizeof(((t *)0)->f))
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#define ALIGN_POW2(value, pow2) \
	(((value) + ((pow2 >> 1) - 1)) / (pow2)) * (pow2)

#define ALLOCATION(ptr, count, func) (func((count) * sizeof(*(ptr))))

#define MALLOC(ptr, count) ALLOCATION(ptr, count, malloc)
#define ALLOCA(ptr, count) ALLOCATION(ptr, count, alloca)
#define REALLOC(ptr, count) (realloc(ptr, (count) * sizeof(*(ptr))))

ALLOC void *alloc_huge_page(size_t size);

size_t available_memory(void);

#endif // SYSTEM_MEMORY_H
