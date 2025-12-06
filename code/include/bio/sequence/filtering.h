#pragma once
#ifndef BIO_SEQUENCE_FILTERING_H
#define BIO_SEQUENCE_FILTERING_H

#include "bio/types.h"

bool arg_mode_filter(void);

bool filter_seqs(sequence_t *seqs, s32 seq_n, bool *kept, s32 *seq_n_filter);

#endif /* BIO_SEQUENCE_FILTERING_H */
