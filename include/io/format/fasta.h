#pragma once
#ifndef IO_FORMAT_FASTA_H
#define IO_FORMAT_FASTA_H

#include <stdbool.h>
#include <stddef.h>

#include "io/input.h"
#include "system/compiler.h"

bool fasta_detect(struct ifile S(1), const char PRS(ext, 1));

bool fasta_open(struct ifile S(1));

size_t fasta_entry_count(struct ifile S(1));

size_t fasta_entry_length(struct ifile S(1));

size_t fasta_entry_extract(struct ifile S(1), char RS(1));

bool fasta_entry_next(struct ifile S(1));

#endif /* IO_FORMAT_FASTA_H */
