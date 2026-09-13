#ifndef BIO_METHOD_H
#define BIO_METHOD_H

#include "system/types.h"

struct meta {
	uhz off;
	uhz len;
};

constexpr uhz SEQ_LUT_SIZE = 1 << 7;
extern shz SEQ_LUT[SEQ_LUT_SIZE];
constexpr uhz SUB_MAT_DIM = 24;
extern shz SUB_MAT[SUB_MAT_DIM][SUB_MAT_DIM];

extern shz GAP_PEN;
extern shz GAP_OPN;
extern shz GAP_EXT;
constexpr shz SCORE_MIN = SHZ_MIN / 2;

constexpr uhz SEQ_N_MIN = 2;
constexpr uhz SEQ_LEN_MIN = 1;
constexpr uhz SEQ_LEN_MAX = UHZ_MAX - 1;

#define LEN_BAD(l) (l < SEQ_LEN_MIN || l > SEQ_LEN_MAX)
#define SEQ_BAD(s) (!*s)

extern usz TABLE_SIZE;

extern const struct methods {
	shz (*const method)(uhz, uhz, shz *restrict, const u8 *restrict);
	struct arg_callback (*const validate)(void);
	const char *kernel;
	const char *name;
	const char *arg;
	const enum { GAP_LINEAR, GAP_AFFINE } gap;
} __start_methods[], __stop_methods[], *ALIGN;

#define ALIGN_REGISTER(NAME, ARG, GAP)                                  \
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
