#ifndef SYSTEM_OS_H
#define SYSTEM_OS_H

#include <stddef.h>

#include "util/macros.h"

constexpr size_t KiB = 1 << 10;
constexpr size_t MiB = KiB << 10;
constexpr size_t GiB = MiB << 10;

constexpr size_t CACHE_LINE = 64;

#define MALLOCA(ptr, elem) ptr = malloc(bytesof(ptr, (elem)))
#define MALLOCA_AL(ptr, al, elem) ptr = alloc_aligned(al, bytesof(ptr, (elem)))

size_t memory_cpu(void);
size_t memory_gpu(void);

void free_aligned(void *ptr);

[[gnu::malloc, gnu::malloc(free_aligned, 1), gnu::alloc_align(1),
  gnu::alloc_size(2)]]
void *alloc_aligned(size_t alignment, size_t bytes);

void free_mmap(void *alloced_mmap);

[[gnu::malloc, gnu::malloc(free_mmap, 1), gnu::alloc_size(1)]]
void *alloc_mmap(size_t bytes, bool tmpfile);

[[gnu::malloc, gnu::malloc(free_aligned, 1), gnu::alloc_align(3)]]
void *copy_file(const char *path, void **end, size_t alignment);

extern int THREAD_NUM;

double time_current(void);

void *dll_open(const char *name);
void *dll_sym(void *restrict, const char *restrict symbol);
bool dll_close(void *);

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
