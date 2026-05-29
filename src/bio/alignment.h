#ifndef BIO_ALIGNMENT_H
#define BIO_ALIGNMENT_H

#include "system/types.h"

struct input;
struct output;
struct sequence;

[[gnu::nonnull]]
bool align(const struct input *, struct output *);

constexpr size_t SEQ_LUT_SIZE = 1 << 7;
extern s32 SEQ_LUT[SEQ_LUT_SIZE];
constexpr size_t SUB_MAT_DIM = 24;
extern s32 SUB_MAT[SUB_MAT_DIM][SUB_MAT_DIM];

extern s32 GAP_PEN;
extern s32 GAP_OPN;
extern s32 GAP_EXT;
constexpr s32 SCORE_MIN = S32_MIN / 2;

extern enum align_methods {
	ALIGN_INVALID = -1,
	ALIGN_GA,
	ALIGN_NW,
	ALIGN_SW,
	ALIGN_COUNT
} METHOD_ID;

extern struct align_method {
	s32 (*method)(const struct sequence *restrict,
		      const struct sequence *restrict, const s32 *restrict,
		      s32 *restrict);
	const char *name;
	const char **aliases;
	enum {
		GAP_LINEAR,
		GAP_AFFINE,
	} gap;
} ALIGN_METHODS[ALIGN_COUNT];

#define ALIGN_METHOD(ID, FN, GAP, NAME, ALIAS, ...)                            \
	[[gnu::constructor]]                                                   \
	static void ID##_REGISTER(void)                                        \
	{                                                                      \
		static_assert(ID > ALIGN_INVALID && ID < ALIGN_COUNT);         \
		static const char *a[] = { ALIAS, ##__VA_ARGS__, nullptr };    \
		ALIGN_METHODS[ID] = (struct align_method){ FN, NAME, a, GAP }; \
	}

#endif /* BIO_ALIGNMENT_H */
