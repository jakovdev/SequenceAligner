#pragma once
#ifndef IO_INPUT_H
#define IO_INPUT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

enum input_format {
	INPUT_FORMAT_UNKNOWN,
	INPUT_FORMAT_FASTA,
	INPUT_FORMAT_DSV,
};

struct dsv_context {
	size_t target_column;
	size_t num_columns;
	char delimiter;
};

struct fasta_context {
	int reserved;
};

struct ifile {
	FILE *stream;
	char *line;
	size_t line_cap;
	size_t entries;
	enum input_format format;
	union {
		struct dsv_context dsv;
		struct fasta_context fasta;
	} ctx;
};

bool ifile_open(struct ifile *, const char *restrict path);

void ifile_close(struct ifile *);

size_t ifile_entry_total(struct ifile *);

void ifile_entry_length(struct ifile *, size_t *length);

void ifile_entry_extract(struct ifile *, char *restrict output, size_t length);

bool ifile_entry_next(struct ifile *);

const char *arg_input(void);

#endif /* IO_INPUT_H */
