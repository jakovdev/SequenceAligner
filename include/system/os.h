#ifndef SYSTEM_OS_H
#define SYSTEM_OS_H

#include <stddef.h>

#include "util/macros.h"

constexpr size_t KiB = 1 << 10;
constexpr size_t MiB = KiB << 10;
constexpr size_t GiB = MiB << 10;

constexpr size_t CACHE_LINE = 64;
constexpr size_t PAGE_SIZE = 4 * KiB;

#define MALLOC(ptr, bytes) ptr = (typeof(ptr))malloc(bytes)
#define MALLOCA(ptr, nmemb) MALLOC(ptr, bytesof(ptr, nmemb))
#define MALLOC_AL(ptr, al, bytes) ptr = (typeof(ptr))alloc_aligned(al, bytes)
#define MALLOCA_AL(ptr, al, nmemb) MALLOC_AL(ptr, al, bytesof(ptr, nmemb))

size_t available_memory(void);

void free_aligned(void *ptr);

[[gnu::malloc, gnu::malloc(free_aligned, 1), gnu::alloc_size(2)]]
void *alloc_aligned(size_t alignment, size_t bytes);

void free_mmap(void *mmap);

[[gnu::malloc, gnu::malloc(free_mmap, 1), gnu::alloc_size(1)]]
void *alloc_mmap(size_t bytes);

extern int THREAD_NUM;

double time_current(void);

[[gnu::nonnull]]
const char *file_name(const char *path);
[[gnu::nonnull]]
bool path_special_exists(const char *path);
[[gnu::nonnull]]
bool path_file_exists(const char *path);
[[gnu::nonnull]]
bool path_directories_create(const char *path);

struct arg_callback parse_path(const char *str, void *dest);

#endif /* SYSTEM_OS_H */
