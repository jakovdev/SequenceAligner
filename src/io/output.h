#ifndef IO_OUTPUT_H
#define IO_OUTPUT_H

#include <stddef.h>

#include "system/types.h"
#include "util/macros.h"

struct input;
struct output {
	s32 *restrict matrix;
	const char **seqs;
	size_t dim;
	bool mmap;
	bool triangular;
};

[[gnu::nonnull]]
bool output_load(struct output *, const struct input *);
[[gnu::nonnull]]
void output_fill(struct output *, const s32 *columns, s32 col);
[[gnu::nonnull]]
bool output_flush(struct output *);
[[gnu::nonnull]]
void output_free(struct output *);

extern enum output_format {
	FLUSH_INVALID = -1,
	FLUSH_HDF5,
	FLUSH_COUNT
} FLUSH_ID;

extern bool (*FLUSH_FORMATS[FLUSH_COUNT])(struct output *, const char *);

#define FLUSH_REGISTER(ID, FN)                                           \
	[[gnu::constructor]]                                             \
	static void ID##_REGISTER(void)                                  \
	{                                                                \
		static_assert(ID > FLUSH_INVALID && ID < FLUSH_COUNT);   \
		static_assert(ARRAY_SIZE(FLUSH_FORMATS) == FLUSH_COUNT); \
		FLUSH_FORMATS[ID] = FN;                                  \
	}

#endif /* IO_OUTPUT_H */
