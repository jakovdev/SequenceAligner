#pragma once
#ifndef IO_FORMAT_DSV_H
#define IO_FORMAT_DSV_H

#include <stdbool.h>
#include <stddef.h>

struct ifile;

bool dsv_detect(struct ifile *, const char *restrict extension);

bool dsv_open(struct ifile *);

size_t dsv_entry_count(struct ifile *);

void dsv_entry_length(struct ifile *, size_t *length);

void dsv_entry_extract(struct ifile *, char *restrict output, size_t length);

bool dsv_entry_next(struct ifile *);

#endif /* IO_FORMAT_DSV_H */
