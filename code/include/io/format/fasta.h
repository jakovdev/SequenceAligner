#pragma once
#ifndef IO_FORMAT_FASTA_H
#define IO_FORMAT_FASTA_H

#include <stdbool.h>
#include <stddef.h>

#include "system/types.h"

bool fasta_entry_next(char *restrict *restrict p_cursor);

u64 fasta_total_entries(char *restrict file_cursor, char *restrict file_end);

u64 fasta_entry_extract(char *restrict *restrict p_cursor,
			char *restrict file_end, char *restrict output);

u64 fasta_entry_length(char *cursor, char *file_end);

bool fasta_validate(char *restrict file_start, char *restrict file_end);

#endif // IO_FORMAT_FASTA_H
