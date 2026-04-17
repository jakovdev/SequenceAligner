#pragma once
#ifndef BIO_ALGORITHM_METHOD_GA_H
#define BIO_ALGORITHM_METHOD_GA_H

#include "bio/sequence/sequences.h"

[[gnu::sysv_abi, gnu::noinline, gnu::hot]]
s32 align_ga(SEQ_PTR(), SEQ_PTR(), s32 *restrict TABLE, s32 *restrict SEQ1I);

#endif /* BIO_ALGORITHM_METHOD_GA_H */
