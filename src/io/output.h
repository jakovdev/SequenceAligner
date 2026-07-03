#ifndef IO_OUTPUT_H
#define IO_OUTPUT_H

#include <stddef.h>

#include "system/types.h"

struct input;
struct output {
	s32 *restrict matrix;
	size_t dim;
	const char **seqs;
	bool triangular;
};

[[gnu::nonnull]]
bool output_load(struct output *, struct input);
[[gnu::nonnull]]
void output_fill(struct output, const s32 *cols, size_t col);

bool output_flush(struct output);
[[gnu::nonnull]]
void output_free(struct output *);

#endif /* IO_OUTPUT_H */
