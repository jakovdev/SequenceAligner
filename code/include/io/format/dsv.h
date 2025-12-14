#pragma once
#ifndef IO_FORMAT_DSV_H
#define IO_FORMAT_DSV_H

#include <stdbool.h>
#include <stddef.h>

struct ifile;

bool dsv_detect(struct ifile *, const char *restrict extension);

bool dsv_open(struct ifile *);

size_t dsv_sequence_count(struct ifile *);

size_t dsv_sequence_length(struct ifile *);

size_t dsv_sequence_extract(struct ifile *, char *restrict output);

bool dsv_sequence_next(struct ifile *);

#endif /* IO_FORMAT_DSV_H */
