#pragma once
#ifndef IO_MMAP_H
#define IO_MMAP_H

#include <stdbool.h>
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

void mmap_matrix_name(char *restrict buffer, size_t buffer_size,
		      const char *restrict path);

bool mmap_matrix_open(struct MMapMatrix *, const char *restrict name,
		      size_t dim);

void mmap_matrix_close(struct MMapMatrix *);

s64 matrix_index(s32 row, s32 col);

#endif /* IO_MMAP_H */
