#pragma once
#ifndef UTIL_PROGRESS_H
#define UTIL_PROGRESS_H

#include <stdatomic.h>
#include <stdbool.h>

#include "system/types.h"

bool progress_start(_Atomic(u64) *progress, u64 total, const char *message);

void progress_end(void);

#endif // UTIL_PROGRESS_H
