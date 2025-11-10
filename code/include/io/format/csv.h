#pragma once
#ifndef IO_FORMAT_CSV_H
#define IO_FORMAT_CSV_H

#include <stdbool.h>
#include <stddef.h>

#include "system/types.h"

char *csv_header_parse(char *restrict file_cursor, char *restrict file_end,
		       bool *no_header, u64 *seq_col);

bool csv_line_next(char *restrict *restrict p_cursor);

u64 csv_total_lines(char *restrict file_cursor, char *restrict file_end);

u64 csv_line_column_extract(char *restrict *restrict p_cursor,
			    char *restrict output, u64 target_column);

u64 csv_line_column_length(char *cursor, u64 target_column);

bool csv_validate(const char *restrict file_start,
		  const char *restrict file_end);

#endif // IO_FORMAT_CSV_H
