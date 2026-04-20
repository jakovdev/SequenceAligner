#ifndef SYSTEM_OS_H
#define SYSTEM_OS_H

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else /* POSIX/Linux */

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <sys/stat.h>

#ifndef PATH_MAX
#define PATH_MAX _POSIX_PATH_MAX
#endif
#define MAX_PATH PATH_MAX

#endif

extern int THREAD_NUM;
struct arg_callback parse_path(const char *str, void *dest);

double time_current(void);

[[gnu::nonnull]]
const char *file_name(const char *path);
[[gnu::nonnull]]
bool path_special_exists(const char *path);
[[gnu::nonnull]]
bool path_file_exists(const char *path);
[[gnu::nonnull]]
bool path_directories_create(const char *path);

#endif /* SYSTEM_OS_H */
