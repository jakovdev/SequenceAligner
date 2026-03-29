#pragma once
#ifndef BIO_ALGORITHM_ALIGNMENT_CUH
#define BIO_ALGORITHM_ALIGNMENT_CUH

#ifdef USE_CUDA

#ifdef __cplusplus

#if __cplusplus < 201703L
#include <cstdbool>
#endif

#include <climits>

extern "C" {

#else /* C */
#include <stdbool.h>
#include <limits.h>
#define __restrict__ restrict
#endif /* __cplusplus */

#include <driver_types.h>

#include "system/types.h"

#define MAX_CUDA_SEQUENCE_LENGTH (1023)
#define SUB_MATDIM (24)

struct Constants {
	char *letters;
	s32 *lengths;
	s64 *offsets;
	sll *progress;
	sll *checksum;
	s32 seq_lup[SCHAR_MAX + 1];
	s32 sub_mat[SUB_MATDIM * SUB_MATDIM];
	s32 seq_n;
	s32 gap_pen;
	s32 gap_open;
	s32 gap_ext;
	bool triangular;
};

void cuda_config(uint grid_max, uint block_max, cudaStream_t stream);
cudaError_t copy_constants(const struct Constants *host);

typedef cudaError_t (*kernel_func_t)(s32 *__restrict__, s64, s64);
cudaError_t kernel_nw(s32 *__restrict__ scores, s64 start, s64 batch);
cudaError_t kernel_ga(s32 *__restrict__ scores, s64 start, s64 batch);
cudaError_t kernel_sw(s32 *__restrict__ scores, s64 start, s64 batch);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* USE_CUDA */

#endif /* BIO_ALGORITHM_ALIGNMENT_CUH */
