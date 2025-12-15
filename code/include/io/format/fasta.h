#pragma once
#ifndef IO_FORMAT_FASTA_H
#define IO_FORMAT_FASTA_H

#include <stdbool.h>
#include <stddef.h>

struct ifile;

bool fasta_detect(struct ifile *, const char *restrict extension);

bool fasta_open(struct ifile *);

size_t fasta_sequence_count(struct ifile *);

void fasta_sequence_length(struct ifile *, size_t *out_length);

void fasta_sequence_extract(struct ifile *, char *restrict output,
			    size_t expected_length);

bool fasta_sequence_next(struct ifile *);

#endif /* IO_FORMAT_FASTA_H */
