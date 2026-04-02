#pragma once
#ifndef IO_FORMAT_DSV_H
#define IO_FORMAT_DSV_H

#include <stdbool.h>
#include <stddef.h>

#include "io/input.h"

bool dsv_detect(struct ifile[static 1], const char ext[restrict static 1]);

bool dsv_open(struct ifile[static 1]);

size_t dsv_entry_count(struct ifile[static 1]);

void dsv_entry_length(struct ifile[static 1], size_t length[static 1]);

void dsv_entry_extract(struct ifile[static 1], char output[restrict static 1],
		       size_t length);

bool dsv_entry_next(struct ifile[static 1]);

#endif /* IO_FORMAT_DSV_H */
