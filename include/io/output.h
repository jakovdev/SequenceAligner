#ifndef IO_OUTPUT_H
#define IO_OUTPUT_H

#include <stddef.h>

#include "system/types.h"
#ifdef _WIN32
#include "system/os.h"
#endif

struct input;

struct mmap {
#ifdef _WIN32
	HANDLE hFile, hMapping;
#else
	int fd;
#endif
};

struct output {
	s32 *restrict matrix;
	size_t bytes;
	const char **seqs;
	size_t dim;
	struct mmap file;
	bool mmap;
	bool triangular;
};

[[gnu::nonnull]]
bool output_load(struct output *, const struct input *);
[[gnu::nonnull]]
void output_fill(struct output *, s32 col, const s32 *columns);
[[gnu::nonnull]]
bool output_flush(struct output *);
[[gnu::nonnull]]
void output_free(struct output *);

extern enum output_format {
	FLUSH_INVALID = -1,
	FLUSH_HDF5,
	FLUSH_COUNT
} FLUSH_ID;

typedef bool (*flush_fn)(struct output *, const char *);
extern flush_fn FLUSH_FORMATS[FLUSH_COUNT];

#include <args.h>

#include "util/macros.h"

#define FLUSH_REGISTER(ID, FN)                                           \
	_ARGS_CONSTRUCTOR(FLUSH_FORMATS_REGISTER_##ID)                   \
	{                                                                \
		static_assert(ID > FLUSH_INVALID && ID < FLUSH_COUNT);   \
		static_assert(ARRAY_SIZE(FLUSH_FORMATS) == FLUSH_COUNT); \
		FLUSH_FORMATS[ID] = FN;                                  \
	}

#endif /* IO_OUTPUT_H */
