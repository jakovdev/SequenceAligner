#ifndef BIO_KERNELS_CUH
#define BIO_KERNELS_CUH

#ifdef __cplusplus
extern "C" {
#endif

#include "bio/method.h"

constexpr uhz MAX_CUDA_SEQUENCE_LENGTH = 1023;

struct constants {
	u8 *letters;
	struct meta *meta;
	shz seq_lut[SEQ_LUT_SIZE];
	shz sub_mat[SUB_MAT_DIM * SUB_MAT_DIM];
	uhz num;
	shz gap_pen;
	shz gap_opn;
	shz gap_ext;
	bool triangular;
};

#ifdef __cplusplus
}
#endif

#endif /* BIO_KERNELS_CUH */
