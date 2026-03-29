#pragma once
#ifndef BIO_ALGORITHM_ALIGNMENT_CUH
#define BIO_ALGORITHM_ALIGNMENT_CUH

#ifdef USE_CUDA

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include <driver_types.h>
#include <stdbool.h>

#include "bio/types.h"

#define MAX_CUDA_SEQUENCE_LENGTH (1023)

struct Constants {
	char *letters;
	s32 *lengths;
	s64 *offsets;
	sll *progress;
	sll *checksum;
	s32 seq_lup[SEQ_LUP_SIZE];
	s32 sub_mat[SUBMAT_MAX * SUBMAT_MAX];
	s32 seq_n;
	s32 gap_pen;
	s32 gap_open;
	s32 gap_ext;
	bool triangular;
};

void cuda_config(uint grid_max, uint block_max, cudaStream_t stream);
cudaError_t copy_constants(const struct Constants *host);
cudaError_t kernel_nw(s32 *scores, s64 start, s64 batch);
cudaError_t kernel_ga(s32 *scores, s64 start, s64 batch);
cudaError_t kernel_sw(s32 *scores, s64 start, s64 batch);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* USE_CUDA */

#endif /* BIO_ALGORITHM_ALIGNMENT_CUH */
