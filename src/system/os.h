#ifndef SYSTEM_OS_H
#define SYSTEM_OS_H

#include "system/types.h"

constexpr usz KiB = 1 << 10;
constexpr usz MiB = KiB << 10;
constexpr usz GiB = MiB << 10;

constexpr usz CACHE_LINE = 64;

usz memory_cpu(void);
usz memory_gpu(void);

void free_al(void *ptr);

[[gnu::malloc, gnu::malloc(free_al, 1), gnu::alloc_align(1),
  gnu::alloc_size(2)]]
void *alloc_al(usz alignment, usz bytes);
#define alloc_cl(bytes) alloc_al(CACHE_LINE, bytes)

void free_mmap(void *alloced_mmap);

[[gnu::malloc, gnu::malloc(free_mmap, 1), gnu::alloc_size(1)]]
void *alloc_mmap(usz bytes, bool tmpfile);

[[gnu::malloc, gnu::malloc(free_al, 1), gnu::alloc_align(3)]]
void *copy_file(const char *path, void **end, usz alignment);

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
