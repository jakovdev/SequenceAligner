#ifndef IO_INPUT_H
#define IO_INPUT_H

#include "system/types.h"

struct input {
	u8 *restrict seqs;
	struct meta *restrict meta;
	uhz max;
	uhz num;
};

[[gnu::nonnull]]
bool input_load(struct input *);
[[gnu::nonnull]]
void input_free(struct input *);

#endif /* IO_INPUT_H */
