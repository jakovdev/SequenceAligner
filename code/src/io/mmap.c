#include "io/mmap.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "system/memory.h"
#include "system/os.h"
#include "util/print.h"

void mmap_matrix_name(char *buffer, size_t buffer_size, const char *path)
{
	if unlikely (!buffer || !buffer_size || !path || !*path) {
		pdev("Invalid parameters for mmap_matrix_name()");
		perr("Internal error generating matrix file name");
		pabort();
	}

	char dir[MAX_PATH] = { 0 };
	char base[MAX_PATH] = { 0 };

	const char *last_slash = strrchr(path, '/');
	if (last_slash) {
		ptrdiff_t delta = last_slash - path + 1;
		size_t dir_len = delta < 0 ? 0 : (size_t)delta;
		strncpy(dir, path, dir_len);
		dir[dir_len] = '\0';
		strncpy(base, last_slash + 1, MAX_PATH - 1);
	} else {
		strcpy(dir, "./");
		strncpy(base, path, MAX_PATH - 1);
	}

	base[MAX_PATH - 1] = '\0';
	char *dot = strrchr(base, '.');
	if (dot)
		*dot = '\0';

	snprintf(buffer, buffer_size, "%s%s.mmap", dir, base);
}

static void mmap_matrix_init(struct MMapMatrix *file)
{
	if (!file)
		unreachable();
#ifdef _WIN32
	file->hFile = INVALID_HANDLE_VALUE;
	file->hMapping = NULL;
#else
	file->fd = -1;
#endif
	file->bytes = 0;
	file->matrix = NULL;
}

bool mmap_matrix_open(struct MMapMatrix *restrict file,
		      const char *restrict name, size_t dim)
{
	if unlikely (!file || !name || dim < 2) {
		pdev("Invalid parameters for mmap_matrix_open()");
		perr("Internal error opening score matrix file");
		pabort();
	}

	mmap_matrix_init(file);
	const size_t bytes = bytesof(file->matrix, dim * (dim - 1) / 2);
	pinfol("Creating matrix file: %s (%.2f GiB)", file_name(name),
	       (double)bytes / (double)GiB);

#define file_error_return(message_lit)                      \
	do {                                                \
		perr(message_lit " '%s'", file_name(name)); \
		mmap_matrix_close(file);                    \
		return false;                               \
	} while (0)

#ifdef _WIN32
	file->hFile = CreateFileA(name, GENERIC_READ | GENERIC_WRITE, 0, NULL,
				  CREATE_ALWAYS, FILE_FLAG_DELETE_ON_CLOSE,
				  NULL);
	if (file->hFile == INVALID_HANDLE_VALUE)
		file_error_return("Could not create memory-mapped file");

	LARGE_INTEGER file_size;
	file_size.QuadPart = (LONGLONG)bytes;
	SetFilePointerEx(file->hFile, file_size, NULL, FILE_BEGIN);
	SetEndOfFile(file->hFile);

	file->hMapping = CreateFileMapping(file->hFile, NULL, PAGE_READWRITE, 0,
					   0, NULL);
	if (!file->hMapping)
		file_error_return("Could not create file mapping for");

	file->matrix =
		MapViewOfFile(file->hMapping, FILE_MAP_ALL_ACCESS, 0, 0, 0);
	if (!file->matrix)
		file_error_return("Could not map view of file");
#else
	file->fd = open(name, O_RDWR | O_CREAT | O_TRUNC, 0644);
	if (file->fd == -1)
		file_error_return("Could not create memory-mapped file");

	if (unlink(name) == -1)
		file_error_return("Could not unlink memory-mapped file");

	if (ftruncate(file->fd, (off_t)bytes) == -1)
		file_error_return("Could not set size for file");

	file->matrix = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED,
			    file->fd, 0);
	if (file->matrix == MAP_FAILED)
		file_error_return("Could not memory map file");

	madvise(file->matrix, bytes, MADV_RANDOM);
	madvise(file->matrix, bytes, MADV_HUGEPAGE);
	madvise(file->matrix, bytes, MADV_DONTFORK);
	madvise(file->matrix, bytes, MADV_DONTDUMP);
#endif
	file->bytes = bytes;
	return true;

#undef file_error_return
}

void mmap_matrix_close(struct MMapMatrix *file)
{
	if unlikely (!file) {
		pdev("NULL file in mmap_matrix_close()");
		perr("Internal error closing score matrix file");
		pabort();
	}

#ifdef _WIN32
	if (file->matrix) {
		UnmapViewOfFile(file->matrix);
		file->matrix = NULL;
	}

	if (file->hMapping) {
		CloseHandle(file->hMapping);
		file->hMapping = NULL;
	}

	if (file->hFile != INVALID_HANDLE_VALUE) {
		CloseHandle(file->hFile);
		file->hFile = INVALID_HANDLE_VALUE;
	}
#else
	if (file->matrix) {
		munmap(file->matrix, file->bytes);
		file->matrix = NULL;
	}

	if (file->fd != -1) {
		close(file->fd);
		file->fd = -1;
	}
#endif
	file->bytes = 0;
}

s64 matrix_index(s32 row, s32 col)
{
	return ((s64)col * (col - 1)) / 2 + row;
}
