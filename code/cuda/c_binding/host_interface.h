#pragma once
#ifndef HOST_INTERFACE_H
#define HOST_INTERFACE_H

#ifndef __cplusplus
#include <stddef.h>
#include <stdbool.h>
#endif

#include "host_types.h"

#ifdef __cplusplus
extern "C" {
#endif

bool cuda_triangular(size_t buffer_bytes);

bool cuda_upload_sequences(const sequence_t *seqs, u32 seq_n, u32 seq_len_max,
			   u64 seq_len_sum);
bool cuda_upload_scoring(s32 sub_mat[SUB_MATDIM][SUB_MATDIM],
			 s32 seq_lup[SEQ_LUPSIZ]);
bool cuda_upload_gaps(s32 linear, s32 start, s32 extend);
bool cuda_upload_storage(s32 *scores, size_t scores_bytes);

bool cuda_kernel_launch(int kernel_id);
bool cuda_kernel_results(void);
ull cuda_kernel_progress(void);
sll cuda_kernel_checksum(void);

const char *cuda_error_device(void);
const char *cuda_error_host(void);
const char *cuda_device_name(void);

#ifdef __cplusplus
}

#endif

#endif /* HOST_INTERFACE_H */
