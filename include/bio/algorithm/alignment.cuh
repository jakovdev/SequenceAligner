#pragma once
#ifndef BIO_ALGORITHM_ALIGNMENT_CUH
#define BIO_ALGORITHM_ALIGNMENT_CUH

#ifdef __cplusplus
extern "C" {
#endif

#include <driver_types.h>

#include "bio/score/matrices.h"
#include "bio/types.h"

#define MAX_CUDA_SEQUENCE_LENGTH (1023)

struct Constants {
	char *letters;
	s32 *lengths;
	s64 *offsets;
	sll *progress;
	sll *checksum;
	s32 seq_lut[SEQ_LUT_SIZE];
	s32 sub_mat[SUB_MAT_DIM * SUB_MAT_DIM];
	s32 seq_n;
	s32 gap_pen;
	s32 gap_open;
	s32 gap_ext;
	bool triangular;
};

[[gnu::nonnull]]
cudaError_t copy_constants(const struct Constants *host);
extern const void *kernels[ALIGN_COUNT];

#ifdef __cplusplus
}
#endif

#endif /* BIO_ALGORITHM_ALIGNMENT_CUH */
