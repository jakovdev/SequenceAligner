#pragma once
#ifndef IO_FILES_H
#define IO_FILES_H

#include <stddef.h>
#include <stdbool.h>

#include "system/types.h"
#ifdef _WIN32
#include "system/os.h"
#endif

struct FileMetadata {
	size_t bytes;
#ifdef _WIN32
	HANDLE hFile;
	HANDLE hMapping;
#else
	int fd;
#endif
};

enum FileFormat { FILE_FORMAT_CSV, FILE_FORMAT_FASTA, FILE_FORMAT_UNKNOWN };

struct FileFormatMetadata {
	char *start;
	char *end;
	char *cursor;
	union {
		struct {
			u64 sequence_column;
			bool headerless;
		} csv;

		struct {
			u64 temp1;
			bool temp2;
		} fasta;
	} format;
	enum FileFormat type;
	u32 total;
};

struct FileText {
	struct FileMetadata meta;
	struct FileFormatMetadata data;
	char *text;
};

struct FileScoreMatrix {
	struct FileMetadata meta;
	s32 *matrix;
};

bool file_text_open(struct FileText *restrict file, const char *restrict path);
void file_text_close(struct FileText *file);

u32 file_sequence_total(struct FileText *file);

u64 file_sequence_next_length(struct FileText *file);

bool file_sequence_next(struct FileText *file);

u64 file_extract_entry(struct FileText *restrict file, char *restrict out);

struct FileScoreMatrix file_matrix_open(const char *path, u64 matrix_dim);
void file_matrix_close(struct FileScoreMatrix *file);

u64 matrix_index(u32 row, u32 col);

void file_matrix_name(char *buffer, size_t buffer_size, const char *path);

#endif // IO_FILES_H
