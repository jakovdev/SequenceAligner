#pragma once
#ifndef BIO_ALGORITHM_METHOD_NW_H
#define BIO_ALGORITHM_METHOD_NW_H

#include "bio/sequence/sequences.h"

[[gnu::nonnull, gnu::noinline, gnu::hot]]
s32 align_nw(seq_ptr, seq_ptr, s32 *restrict TABLE, s32 *restrict SEQ1I);

#endif /* BIO_ALGORITHM_METHOD_NW_H */
