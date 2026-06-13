#include "io/input.h"

#include <args.h>
#include <ctype.h>
#include <limits.h>
#include <print.h>
#include <string.h>

#include "bio/align.h"
#include "io/source.h"
#include "system/os.h"
#include "util/benchmark.h"
#include "util/macros.h"

static const char *INPUT_PATH;
bool filter(struct input *);

static bool sequence_length_limit(s32 len)
{
	s32 gap = -GAP_PEN;
	if (!gap)
		return len <= SEQ_LEN_MAX;
	return len <= SEQ_LEN_MAX / gap;
}

static bool sequence_normalize(uchar *restrict seq, s32 *restrict len)
{
	s32 end = *len;
	s32 cur = 0;
	for (s32 i = 0; i < end; i++) {
		uchar c = (uchar)toupper(seq[i]);
		if (c == '\r' || c == '\n' || c == ' ')
			continue;
		if (c == '\0' || c >= SCHAR_MAX) {
			perr("Possible file corruption found");
			return false;
		}
		if (SEQ_LUT[c] < 0)
			return false;
		seq[cur++] = c;
	}
	*len = cur;
	return true;
}

void input_free(struct input *in)
{
	free_aligned(in->meta);
	free_aligned(in->letters);
	memset(in, 0, sizeof(*in));
}

bool input_load(struct input *in)
{
	bench_input_start();
	const char *path = INPUT_PATH;
	const char *name = file_name(path);
	const char *dot = strrchr(name, '.');
	if (!dot || dot == name) {
		perr("File extension not found: %s", name);
		return false;
	}

	void *begin, *end;
	if (!read_mmap(path, &begin, &end))
		return false;

	struct source src = { .file = begin, .fend = end, .ext = dot + 1 };
	for (auto s = __start_sources; s < __stop_sources; s++) {
		switch (s->parse(&src)) {
		case SOURCE_UNSUPPORTED:
			continue;
		case SOURCE_SUCCESS:
			goto parse_success;
		case SOURCE_ERROR:
			unread_mmap(begin, end);
			return false;
		}
	}

	unread_mmap(begin, end);
	perr("Unsupported file format: %s", name);
	return false;
parse_success:
	if (src.num < SEQ_N_MIN) {
		perr("Not enough sequences: %d (min: %d)", src.num, SEQ_N_MIN);
		return false;
	}
	if (src.num > SEQ_N_MAX) {
		perr("Too many sequences: %d (max: %d)", src.num, SEQ_N_MAX);
		return false;
	}

	input_free(in);
	MALLOCA_AL(in->meta, CACHE_LINE, src.num);
	MALLOCA_AL(in->letters, PAGE_SIZE, src.sum + src.num);
	if (!in->meta || !in->letters) {
		perr("Out of memory for %d sequences", src.num);
		return false;
	}

	s32 large = -1;
	s32 invalid = -1;
	for (s32 i = 0, off = 0; i < src.num; i++) {
#define prompt(var, why)                                            \
	do {                                                        \
		if (var < 0) {                                      \
			bench_input_end();                          \
			pwarn("Sequence #%d " why, i);              \
			var = print_yN("Skip " #var " sequences?"); \
			var--;                                      \
			bench_input_start();                        \
		}                                                   \
		if (var < 0) {                                      \
			perr("Sequence #%d " why, i);               \
			return false;                               \
		}                                                   \
		var++;                                              \
	} while (0)
		struct entry seq = src.entries[i];
		if (!sequence_length_limit(seq.len)) {
			prompt(large, "exceeds length limits");
			continue;
		}

		uchar *restrict dst = in->letters + off;
		memcpy(dst, src.file + seq.off, seq.len);
		dst[seq.len] = '\0';
		if (!sequence_normalize(dst, &seq.len)) {
			prompt(invalid, "has invalid letters");
			continue;
		}
#undef prompt
		in->meta[in->num].len = seq.len;
		in->meta[in->num++].off = off;
		in->max = max(in->max, seq.len);
		off += seq.len + 1;
	}
	free(src.entries);
	unread_mmap(begin, end);
	bench_input_end();

	if (large > 0)
		pinfo("Skipped %d sequences that were too long", large);

	if (invalid > 0)
		pinfo("Skipped %d invalid sequences", invalid);

	if (in->num < SEQ_N_MIN) {
		perr("Not enough valid sequences: %d (min: %d)", in->num,
		     SEQ_N_MIN);
		return false;
	}

	if (!filter(in))
		return false;

	s32 sum = in->meta[in->num - 1].off + in->meta[in->num - 1].len + 1;
	float average_length = (float)sum / (float)in->num - 1.0f;
	pinfo("Loaded %d sequences", in->num);
	pinfo("Average sequence length: %.2f", average_length);
	bench_input_print();
	return true;
}

static void print_input_path(void)
{
	pinfo("Input: %s", file_name(INPUT_PATH));
}

static struct arg_callback validate_input_path(void)
{
	return !path_file_exists(INPUT_PATH) ? ARG_INVALID("File not found") :
					       ARG_VALID();
}

ARGUMENT(input_path) = {
	.dest = &INPUT_PATH,
	.parse_callback = parse_path,
	.validate_callback = validate_input_path,
	.action_callback = print_input_path,
	.arg_req = ARG_REQUIRED,
	.param_req = ARG_PARAM_REQUIRED,
	.validate_phase = ARG_CALLBACK_IF_SET,
	.action_phase = ARG_CALLBACK_ALWAYS,
	.validate_order = ARG_ORDER_FIRST,
	.action_order = ARG_ORDER_FIRST,
	.help_order = ARG_ORDER_FIRST,
	.help = "Input file path: FASTA, DSV (CSV, TSV, etc.) format",
	.param = "FILE",
	.lopt = "input",
	.opt = 'i',
};
