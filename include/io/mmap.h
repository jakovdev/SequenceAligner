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

bool mmap_matrix_open(struct MMapMatrix[static 1], size_t dim);

void mmap_matrix_close(struct MMapMatrix[static 1]);

s64 matrix_index(s32 row, s32 col);

#endif /* IO_MMAP_H */
