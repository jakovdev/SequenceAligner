#pragma once
#ifndef HOST_INTERFACE_H
#define HOST_INTERFACE_H

#include <stdbool.h>

#include "host_types.h"

#ifdef __cplusplus
extern "C" {
#endif

bool cuda_initialize(void);

bool cuda_triangular(size_t buffer_bytes);

bool cuda_upload_sequences(char *sequence_letters, u32 *sequence_offsets,
			   u32 *sequence_lengths, u32 sequence_count,
			   u64 total_sequences_length);

bool cuda_upload_scoring(int *sub_matrix, int *sequence_lookup);

bool cuda_upload_gaps(s32 linear, s32 start, s32 extend);

bool cuda_upload_indices(u64 *indices, s32 *scores, size_t buffer_bytes);

bool cuda_kernel_launch(int kernel_id);

bool cuda_results_get(void);

ull cuda_results_progress(void);
sll cuda_results_checksum(void);

const char *cuda_error_device_get(void);
const char *cuda_error_host_get(void);
const char *cuda_device_name(void);

#ifdef __cplusplus
}

#endif

#endif // HOST_INTERFACE_H
