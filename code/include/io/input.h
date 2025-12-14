#pragma once
#ifndef IO_INPUT_H
#define IO_INPUT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#include "system/types.h"

enum input_format {
	INPUT_FORMAT_UNKNOWN,
	INPUT_FORMAT_FASTA,
	INPUT_FORMAT_DSV,
};

struct dsv_context {
	size_t sequence_column;
	size_t num_columns;
	char delimiter;
};

struct fasta_context {
	int reserved;
};

struct ifile {
	FILE *stream;
	enum input_format format;
	s32 total_sequences;
	union {
		struct dsv_context dsv;
		struct fasta_context fasta;
	} ctx;
};

bool ifile_open(struct ifile *, const char *restrict path);

void ifile_close(struct ifile *);

s32 ifile_sequence_count(struct ifile *);

size_t ifile_sequence_length(struct ifile *);

size_t ifile_sequence_extract(struct ifile *, char *restrict output);

bool ifile_sequence_next(struct ifile *);

const char *arg_input(void);

#endif /* IO_INPUT_H */
