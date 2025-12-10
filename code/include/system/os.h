#pragma once
#ifndef SYSTEM_OS_H
#define SYSTEM_OS_H

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#if defined(_MSC_VER) && !defined(__clang__)
#define strcasecmp _stricmp
#include <Shlwapi.h>
#elif defined(__MINGW32__) || defined(__MINGW64__) || defined(__clang__)
#include <shlwapi.h>
#endif
#define strcasestr StrStrIA
#else /* POSIX/Linux */

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <sys/stat.h>

#define max(a, b) MAX(a, b)
#define min(a, b) MIN(a, b)

#ifndef PATH_MAX
#define PATH_MAX _POSIX_PATH_MAX
#endif
#define MAX_PATH PATH_MAX

#endif

#include <stdatomic.h>
#include <stdbool.h>

#define atomic_load_relaxed(p) atomic_load_explicit((p), memory_order_relaxed)
#define atomic_add_relaxed(p, v) \
	atomic_fetch_add_explicit((p), (v), memory_order_relaxed)

int arg_thread_num(void);

double time_current(void);

const char *file_name_path(const char *path);
bool path_special_exists(const char *path);
bool path_file_exists(const char *path);
bool path_directories_create(const char *path);

#endif /* SYSTEM_OS_H */
