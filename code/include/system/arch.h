#pragma once
#ifndef SYSTEM_ARCH_H
#define SYSTEM_ARCH_H

// GCC/Clang specific macros
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
#else // GCC
#define UNROLL(n) PRAGMA(GCC unroll n)
#define VECTORIZE PRAGMA(GCC ivdep)
#endif // clang
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

// System and memory-related constants
#define KiB (1ULL << 10)
#define MiB (KiB << 10)
#define GiB (MiB << 10)

#define CACHE_LINE ((size_t)64)

#define sizeof_field(t, f) (sizeof(((t *)0)->f))
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#define ALIGN_POW2(value, pow2) \
	(((value) + ((pow2 >> 1) - 1)) / (pow2)) * (pow2)

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>
#if __has_include(<Shlwapi.h>)
#include <Shlwapi.h>
#else // MinGW
#include <shlwapi.h>
#endif
#include <malloc.h>

#ifdef ERROR
#undef ERROR
#endif

typedef HANDLE pthread_t;

typedef DWORD WINAPI T_Func;
#define T_Ret(x) return (DWORD)(size_t)(x)
#define pthread_create(threads, _, function, arg)                             \
	(void)(*threads = CreateThread(NULL, 0,                               \
				       (LPTHREAD_START_ROUTINE)function, arg, \
				       0, NULL))
#define pthread_join(thread_id, _) WaitForSingleObject(thread_id, INFINITE)

#define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)

#define strcasestr(haystack, needle) StrStrIA(haystack, needle)
#define usleep(microseconds) Sleep((microseconds) / 1000)

#define MAX max
#define MIN min

void time_init(void);

#else // POSIX/Linux

#include <alloca.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <unistd.h>

typedef void *T_Func;
#define T_Ret(x) return (x)
#define aligned_free(ptr) free(ptr)
#define time_init()

#ifndef PATH_MAX
#define PATH_MAX _POSIX_PATH_MAX
#endif

#define MAX_PATH PATH_MAX

#endif

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
#error "This project is C only"
#endif

#define ALLOCATION(ptr, count, func) (func((count) * sizeof(*(ptr))))

#define MALLOC(ptr, count) ALLOCATION(ptr, count, malloc)
#define ALLOCA(ptr, count) ALLOCATION(ptr, count, alloca)
#define REALLOC(ptr, count) (realloc(ptr, (count) * sizeof(*(ptr))))

#define atomic_load_relaxed(p) atomic_load_explicit((p), memory_order_relaxed)
#define atomic_add_relaxed(p, v) \
	atomic_fetch_add_explicit((p), (v), memory_order_relaxed)

double time_current(void);

ALLOC void *alloc_huge_page(size_t size);

size_t available_memory(void);

const char *file_name_path(const char *path);

bool path_special_exists(const char *path);
bool path_file_exists(const char *path);
bool path_directories_create(const char *path);

#endif // SYSTEM_ARCH_H
