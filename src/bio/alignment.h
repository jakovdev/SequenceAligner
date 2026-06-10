#ifndef BIO_ALIGNMENT_H
#define BIO_ALIGNMENT_H

#include <args.h>

#include "system/types.h"

struct input;
struct output;
struct sequence;

[[gnu::nonnull]]
bool align(const struct input *, const struct output *);

constexpr s32 SEQ_LUT_SIZE = 1 << 7;
extern s32 SEQ_LUT[SEQ_LUT_SIZE];
constexpr s32 SUB_MAT_DIM = 24;
extern s32 SUB_MAT[SUB_MAT_DIM][SUB_MAT_DIM];

extern s32 GAP_PEN;
extern s32 GAP_OPN;
extern s32 GAP_EXT;
constexpr s32 SCORE_MIN = S32_MIN / 2;

extern const struct align {
	s32 (*const method)(const struct sequence *restrict,
			    const struct sequence *restrict,
			    const s32 *restrict, s32 *restrict);
	struct arg_callback (*const validate)(void);
	const void *const kernel;
	const char **aliases;
	enum {
		GAP_LINEAR,
		GAP_AFFINE,
	} gap;
} __start_aligns[], __stop_aligns[], *ALIGN;

#define ALIGN_REGISTER(NAME)                                \
	[[gnu::aligned(alignof(struct align)), gnu::retain, \
	  gnu::section("aligns"), gnu::used]]               \
	static const struct align __align_##NAME

#define ALIGN_ALIASES(LONG, SHORT, ...) \
	aliases = ((const char *[]){ LONG, SHORT, ##__VA_ARGS__, nullptr })

#ifdef USE_CUDA
#define ALIGN_KERNEL(FN) extern void FN(s32 *, s64, s64)
#else
#define ALIGN_KERNEL(FN) static void *FN = nullptr
#endif

#endif /* BIO_ALIGNMENT_H */
