#pragma once
#ifndef IO_FORMAT_FASTA_H
#define IO_FORMAT_FASTA_H

#include <stdbool.h>
#include <stddef.h>

#include "io/input.h"

bool fasta_detect(struct ifile[static 1], const char ext[restrict static 1]);

bool fasta_open(struct ifile[static 1]);

size_t fasta_entry_count(struct ifile[static 1]);

void fasta_entry_length(struct ifile[static 1], size_t length[static 1]);

void fasta_entry_extract(struct ifile[static 1], char output[restrict static 1],
			 size_t length);

bool fasta_entry_next(struct ifile[static 1]);

#endif /* IO_FORMAT_FASTA_H */
