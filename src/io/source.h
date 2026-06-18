#ifndef IO_SOURCE_H
#define IO_SOURCE_H

#include "system/types.h"

struct input;
struct source {
	uchar *file;
	const uchar *fend;
	const char *ext;
};

enum parse_result { PARSER_SUCCESS, PARSER_ERROR, PARSER_UNSUPPORTED };
extern const struct sources {
	enum parse_result (*const parse)(struct source, struct input *);
} __start_sources[], __stop_sources[];

#define SOURCE_REGISTER(NAME, PARSER)                               \
	static const struct sources __source_##NAME __attribute__(( \
		SECTION(struct sources, "sources"))) = { .parse = PARSER }

bool sequence_length_limit(s32 len);

#endif /* IO_SOURCE_H */
