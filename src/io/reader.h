#ifndef IO_READER_H
#define IO_READER_H

#include "io/input.h"

struct reader {
	u8 *file;
	const u8 *fend;
	const char *ext;
};

enum reader_result { READER_SUCCESS, READER_ERROR, READER_UNSUPPORTED };
extern const struct readers {
	enum reader_result (*const read)(struct reader, struct input *);
} __start_readers[], __stop_readers[];

#define READER_REGISTER(NAME, READER)                               \
	static const struct readers __reader_##NAME __attribute__(( \
		SECTION(struct readers, "readers"))) = { .read = READER }

bool sequence_length_limit(s32 len);

#endif /* IO_READER_H */
