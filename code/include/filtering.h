#pragma once
#ifndef FILTERING_H
#define FILTERING_H

#include "biotypes.h"

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

#endif // FILTERING_H