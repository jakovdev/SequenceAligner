#pragma once
#ifndef UTIL_PROGRESS_H
#define UTIL_PROGRESS_H

#include <stdatomic.h>
#include <stdbool.h>

#include "system/types.h"

bool progress_start(_Atomic(s64) *progress, s64 total, const char *message);

void progress_end(void);

#endif /* UTIL_PROGRESS_H */
