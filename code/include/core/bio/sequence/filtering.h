#pragma once
#ifndef CORE_BIO_SEQUENCE_FILTERING_H
#define CORE_BIO_SEQUENCE_FILTERING_H

#include "core/bio/types.h"

extern bool filter_sequences_multithreaded(sequences_t sequences,
                                           sequence_count_t sequence_count,
                                           float filter_threshold,
                                           bool* keep_flags,
                                           sequence_count_t* filtered_count);

extern void filter_sequences_singlethreaded(sequences_t sequences,
                                            sequence_count_t sequence_count,
                                            float filter_threshold,
                                            bool* keep_flags,
                                            sequence_count_t* filtered_count);

#endif // CORE_BIO_SEQUENCE_FILTERING_H