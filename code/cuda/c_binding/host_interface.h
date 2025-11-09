#pragma once
#ifndef HOST_INTERFACE_H
#define HOST_INTERFACE_H

#ifdef __cplusplus
#include <cstddef>
#else
#include <stddef.h>
#include <stdbool.h>
#endif

#include "host_types.h"

#ifdef __cplusplus
extern "C" {
#endif

bool cuda_initialize(void);

bool cuda_triangular(size_t buffer_bytes);

bool cuda_upload_sequences(sequence_t *seqs, u32 seq_n, u64 seq_len_total);

bool cuda_upload_scoring(const int sub_mat[SUB_MATDIM][SUB_MATDIM],
			 const int seq_lup[SEQ_LUPSIZ]);

bool cuda_upload_gaps(s32 linear, s32 start, s32 extend);

bool cuda_upload_indices(u64 *indices, s32 *scores, size_t scores_bytes);

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
