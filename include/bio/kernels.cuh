#ifndef BIO_KERNELS_CUH
#define BIO_KERNELS_CUH

#ifdef __cplusplus
extern "C" {
#endif

#include <driver_types.h>

#include "bio/alignment.h"

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

#endif /* BIO_KERNELS_CUH */
