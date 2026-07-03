#ifndef IO_WRITER_H
#define IO_WRITER_H

#include "io/output.h"

enum writer_result { WRITER_SUCCESS, WRITER_ERROR, WRITER_UNSUPPORTED };
extern const struct writers {
	enum writer_result (*const write)(struct output, const char *);
} __start_writers[], __stop_writers[];

#define WRITER_REGISTER(NAME, WRITER)                               \
	static const struct writers __writer_##NAME __attribute__(( \
		SECTION(struct writers, "writers"))) = { .write = WRITER }

#endif /* IO_OUTPUT_H */
