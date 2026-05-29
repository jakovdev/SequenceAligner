#ifndef IO_INPUT_H
#define IO_INPUT_H

#include "system/types.h"

struct input {
	struct sequence *seqs;
	s32 *lengths;
	s64 *offsets;
	char *letters;
	s64 alignments;
	s32 lengths_max;
	s32 seqs_n;
};

[[gnu::nonnull]]
bool input_load(struct input *);
[[gnu::nonnull]]
void input_free(struct input *);

#endif /* IO_INPUT_H */
