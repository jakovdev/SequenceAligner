#pragma once
#ifndef IO_FORMAT_DSV_H
#define IO_FORMAT_DSV_H

#include <stdbool.h>
#include <stddef.h>

#include "io/input.h"
#include "system/compiler.h"

bool dsv_detect(struct ifile S(1), const char PRS(ext, 1));

bool dsv_open(struct ifile S(1));

size_t dsv_entry_count(struct ifile S(1));

size_t dsv_entry_length(struct ifile S(1));

size_t dsv_entry_extract(struct ifile S(1), char RS(1));

bool dsv_entry_next(struct ifile S(1));

#endif /* IO_FORMAT_DSV_H */
