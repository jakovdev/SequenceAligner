#ifndef BIO_KERNELS_CUH
#define BIO_KERNELS_CUH

#ifdef __cplusplus
extern "C" {
#endif

#include "bio/method.h"

constexpr s32 MAX_CUDA_SEQUENCE_LENGTH = 1023;

struct constants {
	u8 *letters;
	struct meta *meta;
	s32 seq_lut[SEQ_LUT_SIZE];
	s32 sub_mat[SUB_MAT_DIM * SUB_MAT_DIM];
	s32 num;
	s32 gap_pen;
	s32 gap_open;
	s32 gap_ext;
	bool triangular;
};

extern const void *const pC;

#ifdef __cplusplus
}
#endif

#endif /* BIO_KERNELS_CUH */
