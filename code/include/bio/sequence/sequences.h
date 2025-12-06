#pragma once
#ifndef BIO_SEQUENCE_SEQUENCES_H
#define BIO_SEQUENCE_SEQUENCES_H

#include "bio/types.h"

bool sequences_load_from_file(void);

sequence_t *sequences_seqs(void);

s32 sequences_seq_n(void);

s64 sequences_alignments(void);

sequence_t *sequence(s32 index);

s32 sequences_seq_len_max(void);

#ifdef USE_CUDA
s64 sequences_seq_len_sum(void);
#endif

#endif /* BIO_SEQUENCE_SEQUENCES_H */
