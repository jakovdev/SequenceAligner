#pragma once
#ifndef UTIL_PROGRESS_H
#define UTIL_PROGRESS_H

#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>

bool progress_start(_Atomic(size_t) *progress, size_t total,
		    const char *message);

void progress_end(void);

#endif // UTIL_PROGRESS_H
