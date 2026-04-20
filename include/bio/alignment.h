#ifndef BIO_ALIGNMENT_H
#define BIO_ALIGNMENT_H

#include <limits.h>

#include "bio/sequences.h"
#include "util/macros.h"

[[gnu::nonnull]]
bool filter(struct input *);
[[gnu::nonnull]]
bool align(const struct input *);

#define SEQ_LUT_SIZE (SCHAR_MAX + 1)
extern s32 SEQ_LUT[SEQ_LUT_SIZE];
#define SUB_MAT_DIM (24)
extern s32 SUB_MAT[SUB_MAT_DIM][SUB_MAT_DIM];

extern s32 GAP_PEN;
extern s32 GAP_OPEN;
extern s32 GAP_EXT;
#define SCORE_MIN (INT32_MIN / 2)

extern enum align_method {
	ALIGN_INVALID = -1,
	ALIGN_GA,
	ALIGN_NW,
	ALIGN_SW,
	ALIGN_COUNT
} METHOD_ID;

enum gap_type {
	GAP_LINEAR,
	GAP_AFFINE,
};

typedef s32 (*align_fn)(seq_ptr, seq_ptr, s32 *restrict, s32 *restrict);
extern align_fn ALIGN_METHODS[ALIGN_COUNT];
extern const char *ALIGN_NAMES[ALIGN_COUNT];
extern const char **ALIGN_ALIASES[ALIGN_COUNT];
extern enum gap_type ALIGN_GAPS[ALIGN_COUNT];

#include <args.h>

#define ALIGN_REGISTER(ALIGN_ARRAY, ID, FN)                            \
	_ARGS_CONSTRUCTOR(ALIGN_ARRAY##_REGISTER_##ID)                 \
	{                                                              \
		static_assert(ID > ALIGN_INVALID && ID < ALIGN_COUNT); \
		static_assert(ARRAY_SIZE(ALIGN_ARRAY) == ALIGN_COUNT); \
		ALIGN_ARRAY[ID] = FN;                                  \
	}

#define ALIGN_METHOD(ID, FN, GAP, NAME, ...)                                  \
	_ARGS_CONSTRUCTOR(REGISTER_##ID)                                      \
	{                                                                     \
		static_assert(ID > ALIGN_INVALID && ID < ALIGN_COUNT);        \
		ALIGN_METHODS[ID] = FN;                                       \
		ALIGN_GAPS[ID] = GAP;                                         \
		ALIGN_NAMES[ID] = NAME;                                       \
		static const char *ID##_ALIASES[] = { __VA_ARGS__, nullptr }; \
		ALIGN_ALIASES[ID] = ID##_ALIASES;                             \
	}

#endif /* BIO_ALIGNMENT_H */
