#ifndef BIO_ALGORITHM_METHOD_SW_H
#define BIO_ALGORITHM_METHOD_SW_H

#include "bio/sequence/sequences.h"

[[gnu::nonnull, gnu::noinline, gnu::hot]]
s32 align_sw(seq_ptr, seq_ptr, s32 *restrict TABLE, s32 *restrict SEQ1I);

#endif /* BIO_ALGORITHM_METHOD_SW_H */
