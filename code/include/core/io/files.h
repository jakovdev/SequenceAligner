#pragma once
#ifndef CORE_IO_FILES_H
#define CORE_IO_FILES_H

#include "core/bio/types.h"

#ifdef _WIN32
#include "system/arch.h"
#endif

typedef struct
{
    size_t bytes;
#ifdef _WIN32
    HANDLE hFile;
    HANDLE hMapping;
#else
    int fd;
#endif
} FileMetadata;

typedef enum
{
    FILE_FORMAT_CSV,
    FILE_FORMAT_FASTA,
    FILE_FORMAT_UNKNOWN
} FileFormat;

typedef struct
{
    sequence_count_t total;
    FileFormat type;
    char* start;
    char* end;
    char* cursor;

    union
    {
        struct
        {
            size_t sequence_column;
            bool headerless;
        } csv;

        struct
        {
            size_t temp1;
            bool temp2;
        } fasta;
    } format;
} FileFormatMetadata;

typedef struct
{
    FileMetadata meta;
    FileFormatMetadata data;
    char* text;
} FileText;

typedef struct
{
    FileMetadata meta;
    score_t* matrix;
} FileScoreMatrix;

bool file_text_open(FileText* restrict file, const char* restrict file_path);
void file_text_close(FileText* file);
sequence_count_t file_sequence_total(FileText* file);
size_t file_sequence_next_length(FileText* file);
bool file_sequence_next(FileText* file);
size_t file_extract_sequence(FileText* restrict file, char* restrict output);
FileScoreMatrix file_matrix_open(const char* file_path, size_t matrix_dim);
void file_matrix_close(FileScoreMatrix* file);
alignment_size_t matrix_triangle_index(sequence_index_t row, sequence_index_t col);
void file_matrix_name(char* buffer, size_t buffer_size, const char* output_path);

#endif // CORE_IO_FILES_H
