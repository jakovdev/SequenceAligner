#ifndef IO_OUTPUT_H
#define IO_OUTPUT_H

#include "system/types.h"

struct input;
struct output {
	shz *restrict matrix;
	usz dim;
	bool triangular;
};

[[gnu::nonnull]]
bool output_load(struct output *, struct input);
[[gnu::nonnull]]
void output_fill(struct output, const shz *cols, usz col);

bool output_flush(struct output, struct input);
[[gnu::nonnull]]
void output_free(struct output *);

#endif /* IO_OUTPUT_H */
