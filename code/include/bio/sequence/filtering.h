#pragma once
#ifndef BIO_SEQUENCE_FILTERING_H
#define BIO_SEQUENCE_FILTERING_H

#include "bio/types.h"
#include "system/types.h"

bool arg_mode_filter(void);

bool filter_sequences(sequence_t *sequences, u32 sequence_count,
		      bool *keep_flags, u32 *filtered_count);

#endif /* BIO_SEQUENCE_FILTERING_H */
