#ifndef IO_INPUT_H
#define IO_INPUT_H

#include "system/types.h"

struct input {
	uchar *restrict letters;
	struct {
		s32 len;
		s32 off;
	} *restrict meta;
	s32 max;
	s32 num;
};

[[gnu::nonnull]]
bool input_load(struct input *);
[[gnu::nonnull]]
void input_free(struct input *);

#endif /* IO_INPUT_H */
