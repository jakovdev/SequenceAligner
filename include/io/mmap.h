#pragma once
#ifndef IO_MMAP_H
#define IO_MMAP_H

#include <stddef.h>

#include "system/types.h"
#ifdef _WIN32
#include "system/os.h"
#endif

struct MMapMatrix {
	size_t bytes;
#ifdef _WIN32
	HANDLE hFile;
	HANDLE hMapping;
#else
	int fd;
#endif
	s32 *matrix;
};

[[gnu::nonnull]]
bool mmap_matrix_open(struct MMapMatrix *, size_t dim);

[[gnu::nonnull]]
void mmap_matrix_close(struct MMapMatrix *);

s64 matrix_index(s32 row, s32 col);

#endif /* IO_MMAP_H */
