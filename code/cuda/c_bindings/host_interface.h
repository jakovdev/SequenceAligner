#pragma once
#ifndef HOST_INTERFACE_H
#define HOST_INTERFACE_H

#include "host_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    bool cuda_initialize(void);
    bool cuda_triangular(size_t buffer_bytes);

    bool cuda_upload_sequences(char* seqs, half_t* offs, half_t* lens, half_t n_sqs, size_t n_chrs);
    bool cuda_upload_scoring(int* scoring_matrix, int* sequence_lookup);
    bool cuda_upload_penalties(int linear, int start, int extend);
    bool cuda_upload_triangle_indices_32(half_t* triangle_indices, int* buffer, size_t buffer_size);
    bool cuda_upload_triangle_indices_64(size_t* triangle_indices, int* buffer, size_t buffer_size);

    bool cuda_kernel_launch(int kernel_id);
    bool cuda_results_get();

    ull cuda_results_progress(void);
    sll cuda_results_checksum(void);

    const char* cuda_error_device_get(void);
    const char* cuda_error_host_get(void);
    const char* cuda_device_name(void);

#ifdef __cplusplus
}

#endif

#endif // HOST_INTERFACE_H