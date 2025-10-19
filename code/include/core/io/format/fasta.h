#pragma once
#ifndef CORE_IO_FORMAT_FASTA_H
#define CORE_IO_FORMAT_FASTA_H

#include <stdbool.h>
#include <stddef.h>

bool fasta_entry_next(char* restrict* restrict p_cursor);

size_t fasta_total_entries(char* restrict file_cursor, char* restrict file_end);

size_t fasta_entry_extract(char* restrict* restrict p_cursor,
                           char* restrict file_end,
                           char* restrict output);

size_t fasta_entry_length(char* cursor, char* file_end);

bool fasta_validate(char* restrict file_start, char* restrict file_end);

#endif // CORE_IO_FORMAT_FASTA_H
