#pragma once
#ifndef CORE_IO_FORMAT_CSV_H
#define CORE_IO_FORMAT_CSV_H

#include <stdbool.h>
#include <stddef.h>

char* csv_header_parse(char* restrict file_cursor,
                       char* restrict file_end,
                       bool* no_header,
                       size_t* seq_col);

bool csv_line_next(char* restrict* restrict p_cursor);

size_t csv_total_lines(char* restrict file_cursor, char* restrict file_end);

size_t csv_line_column_extract(char* restrict* restrict p_cursor,
                               char* restrict output,
                               size_t target_column);

size_t csv_line_column_length(char* cursor, size_t target_column);

bool csv_validate(const char* restrict file_start, const char* restrict file_end);

#endif // CORE_IO_FORMAT_CSV_H