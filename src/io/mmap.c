#include "io/mmap.h"

#include <print.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "system/memory.h"
#include "system/os.h"

static void mmap_matrix_init(struct MMapMatrix PS(file, 1))
{
#ifdef _WIN32
	file->hFile = INVALID_HANDLE_VALUE;
	file->hMapping = NULL;
#else
	file->fd = -1;
#endif
	file->bytes = 0;
	file->matrix = NULL;
}

bool mmap_matrix_open(struct MMapMatrix PS(file, 1), size_t dim)
{
	if unlikely (dim < 2) {
		pdev("Invalid parameters for mmap_matrix_open()");
		perr("Internal error opening score matrix file");
		pabort();
	}

	mmap_matrix_init(file);
	const size_t bytes = bytesof(file->matrix, dim * (dim - 1) / 2);
	pinfol("Creating temporary matrix file (%.2f GiB)",
	       (double)bytes / (double)GiB);

#define file_error_return(message_lit)                      \
	do {                                                \
		perr(message_lit " '%s'", file_name(name)); \
		mmap_matrix_close(file);                    \
		return false;                               \
	} while (0)

#ifdef _WIN32
	char dir[MAX_PATH] = { 0 };
	char name[MAX_PATH] = "temporary matrix file";

	DWORD dir_len = GetTempPathA(MAX_PATH, dir);
	if (!dir_len || dir_len >= MAX_PATH)
		file_error_return("Could not resolve temp directory for");

	if (!GetTempFileNameA(dir, "sqa", 0, name))
		file_error_return("Could not create temp file name for");

	file->hFile = CreateFileA(
		name, GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_ALWAYS,
		FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, NULL);
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
	char name[] = "/tmp/seqalign-mmap-XXXXXX";
	file->fd = mkstemp(name);
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

void mmap_matrix_close(struct MMapMatrix PS(file, 1))
{
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
