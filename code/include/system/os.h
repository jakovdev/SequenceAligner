#pragma once
#ifndef SYSTEM_OS_H
#define SYSTEM_OS_H

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
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

#include <stdbool.h>

int arg_threads(void);

double time_current(void);

const char *file_name_path(const char *path);
bool path_special_exists(const char *path);
bool path_file_exists(const char *path);
bool path_directories_create(const char *path);

#endif /* SYSTEM_OS_H */
