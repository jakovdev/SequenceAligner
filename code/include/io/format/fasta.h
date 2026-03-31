#pragma once
#ifndef IO_FORMAT_FASTA_H
#define IO_FORMAT_FASTA_H

#include <stdbool.h>
#include <stddef.h>

struct ifile;

bool fasta_detect(struct ifile *, const char *restrict extension);

bool fasta_open(struct ifile *);

size_t fasta_entry_count(struct ifile *);

void fasta_entry_length(struct ifile *, size_t *length);

void fasta_entry_extract(struct ifile *, char *restrict output, size_t length);

bool fasta_entry_next(struct ifile *);

#endif /* IO_FORMAT_FASTA_H */
