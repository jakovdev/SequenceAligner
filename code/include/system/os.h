#pragma once
#ifndef SYSTEM_OS_H
#define SYSTEM_OS_H

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>
#if __has_include(<Shlwapi.h>)
#include <Shlwapi.h>
#else /* MinGW */
#include <shlwapi.h>
#endif

typedef HANDLE pthread_t;

#define T_Func DWORD WINAPI
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

void time_init(void);

#else /* POSIX/Linux */

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
#define max MAX
#define min MIN

#ifndef PATH_MAX
#define PATH_MAX _POSIX_PATH_MAX
#endif

#define MAX_PATH PATH_MAX

#endif

#include <stdbool.h>

#define atomic_load_relaxed(p) atomic_load_explicit((p), memory_order_relaxed)
#define atomic_add_relaxed(p, v) \
	atomic_fetch_add_explicit((p), (v), memory_order_relaxed)

double time_current(void);

const char *file_name_path(const char *path);
bool path_special_exists(const char *path);
bool path_file_exists(const char *path);
bool path_directories_create(const char *path);

#endif /* SYSTEM_OS_H */
