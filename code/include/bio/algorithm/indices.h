#pragma once
#ifndef BIO_ALGORITHM_INDICES_H
#define BIO_ALGORITHM_INDICES_H

#include "system/compiler.h"
#include "bio/types.h"

extern thread_local s32 *g_restrict SEQ1I;

void indices_buffers_init(void);

void indices_buffers_free(void);

void indices_precompute(SEQUENCE_PTR_T());

#endif /* BIO_ALGORITHM_INDICES_H */
