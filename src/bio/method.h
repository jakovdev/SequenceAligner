#ifndef BIO_METHOD_H
#define BIO_METHOD_H

#include "system/types.h"

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
constexpr s32 SEQ_LEN_MAX = S32_MAX - 1;

#define LEN_BAD(l) (l < SEQ_LEN_MIN || l > SEQ_LEN_MAX)
#define SEQ_BAD(s) (!*s)

extern size_t TABLE_SIZE;

extern const struct methods {
	s32 (*const method)(s32, s32, s32 *restrict, const u8 *restrict);
	struct arg_callback (*const validate)(void);
	const char *kernel;
	const char *name;
	const char *arg;
	const enum { GAP_LINEAR, GAP_AFFINE } gap;
} __start_methods[], __stop_methods[], *ALIGN;

#define ALIGN_REGISTER(NAME, ARG, GAP)                                  \
	ROSTRING_EXTEND(alignh, "  " NAME ": " #ARG "\\n");             \
	[[gnu::weak]] struct arg_callback validate_##ARG(void);         \
	static const struct methods __method_##ARG                      \
		__attribute__((SECTION(struct methods, "methods"))) = { \
			.method = align_##ARG,                          \
			.validate = validate_##ARG,                     \
			.kernel = "kernel_" #ARG,                       \
			.arg = #ARG,                                    \
			.name = NAME,                                   \
			.gap = GAP,                                     \
		}

#endif /* BIO_METHOD_H */
