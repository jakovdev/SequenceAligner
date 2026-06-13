#ifndef IO_SOURCE_H
#define IO_SOURCE_H

#include "system/types.h"

struct source {
	const uchar *file;
	const uchar *fend;
	const char *ext;
	struct entry {
		s32 off;
		s32 len;
	} *restrict entries;
	s32 num;
	s32 sum;
};

enum source_result { SOURCE_SUCCESS, SOURCE_ERROR, SOURCE_UNSUPPORTED };
extern const struct sources {
	enum source_result (*const parse)(struct source *);
} __start_sources[], __stop_sources[];

#define SOURCE_REGISTER(NAME, PARSER)                               \
	static const struct sources __source_##NAME __attribute__(( \
		SECTION(struct sources, "sources"))) = { .parse = PARSER }

#endif /* IO_SOURCE_H */
