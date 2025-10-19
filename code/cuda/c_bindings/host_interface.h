#pragma once
#ifndef HOST_INTERFACE_H
#define HOST_INTERFACE_H

#include <stdbool.h>

#include "host_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    bool cuda_initialize(void);
    bool cuda_triangular(size_t buffer_bytes);

    bool cuda_upload_sequences(char* sequence_letters,
                               sequence_offset_t* sequence_offsets,
                               quar_t* sequence_lengths,
                               sequence_count_t sequence_count,
                               size_t total_sequences_length);
    bool cuda_upload_scoring(int* scoring_matrix, int* sequence_lookup);
    bool cuda_upload_penalties(int linear, int start, int extend);

    bool cuda_upload_triangle_indices(size_t* triangle_indices,
                                      score_t* score_buffer,
                                      size_t buffer_bytes);

    bool cuda_kernel_launch(int kernel_id);
    bool cuda_results_get(void);

    ull cuda_results_progress(void);
    sll cuda_results_checksum(void);

    const char* cuda_error_device_get(void);
    const char* cuda_error_host_get(void);
    const char* cuda_device_name(void);

#ifdef __cplusplus
}

#endif

#endif // HOST_INTERFACE_H
