#ifndef BIO_ALIGN_H
#define BIO_ALIGN_H

#include "system/types.h"

typedef const uchar *restrict seq;
struct meta {
	s32 off;
	s32 len;
};

constexpr s32 SEQ_LUT_SIZE = 1 << 7;
extern s32 SEQ_LUT[SEQ_LUT_SIZE];
constexpr s32 SUB_MAT_DIM = 24;
extern s32 SUB_MAT[SUB_MAT_DIM][SUB_MAT_DIM];

extern s32 GAP_PEN;
extern s32 GAP_OPN;
extern s32 GAP_EXT;
constexpr s32 SCORE_MIN = S32_MIN / 2;

constexpr s32 SEQ_N_MIN = 2;
constexpr s32 SEQ_LEN_MIN = 1;
constexpr s32 SEQ_LEN_MAX = (S32_MAX - 1) / SEQ_N_MIN;

#define LEN_BAD(l) (l < SEQ_LEN_MIN || l > SEQ_LEN_MAX)
#define SEQ_BAD(s) (!*s)

extern const struct align {
	s32 (*const method)(s32, s32, seq, const s32 *restrict, s32 *restrict);
	struct arg_callback (*const validate)(void);
	const void *const kernel;
	const char **aliases;
	enum {
		GAP_LINEAR,
		GAP_AFFINE,
	} gap;
} __start_aligns[], __stop_aligns[], *ALIGN;

#define ALIGN_REGISTER(NAME)                     \
	static const struct align __align_##NAME \
		__attribute__((SECTION(struct align, "aligns")))

#define ALIGN_ALIASES(LONG, SHORT, ...) \
	aliases = ((const char *[]){ LONG, SHORT, ##__VA_ARGS__, nullptr })

#ifdef USE_CUDA
#define ALIGN_KERNEL(FN) extern void FN(s32 *, s64, s64)
#else
#define ALIGN_KERNEL(FN) constexpr void *FN = nullptr
#endif

#endif /* BIO_ALIGN_H */
