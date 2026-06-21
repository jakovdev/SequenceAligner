#include "io/input.h"

#include <args.h>
#include <print.h>
#include <string.h>

#include "bio/align.h"
#include "io/source.h"
#include "system/os.h"
#include "util/benchmark.h"

static const char *INPUT_PATH;
bool filter(struct input *);

bool sequence_length_limit(s32 len)
{
	s32 gap = -GAP_PEN;
	if (!gap)
		return len <= SEQ_LEN_MAX;
	return len <= SEQ_LEN_MAX / gap;
}

void input_free(struct input *in)
{
	free_aligned(in->meta);
	free_aligned(in->seqs);
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

	pverb("Copying %s into memory", name);
	void *fend;
	uchar *file = copy_file(path, &fend, CACHE_LINE);
	if (!file || (uchar *)fend - file > S32_MAX)
		return false;

	pverbm("Trying out parsers for %s", name);
	for (auto s = __start_sources; s < __stop_sources; s++) {
		switch (s->parse((struct source){ file, fend, dot + 1 }, in)) {
		case PARSER_UNSUPPORTED:
			continue;
		case PARSER_SUCCESS:
			goto parse_success;
		case PARSER_ERROR:
			free_aligned(file);
			return false;
		}
	}

	free_aligned(file);
	perr("Unsupported file format: %s", name);
	return false;
parse_success:
	s32 num = in->num;
	if (num < SEQ_N_MIN) {
		perr("Not enough sequences: %d (min: %d)", num, SEQ_N_MIN);
		return false;
	}

	struct meta *MALLOCA_AL(meta, CACHE_LINE, num);
	if (!meta) {
		perr("Out of memory for %d sequences", num);
		return false;
	}

	in->seqs = file;
	const uchar *p = file;
	for (s32 i = 0; i < num; i++) {
		s32 len = (s32)strlen((const char *)p);
		meta[i] = (struct meta){ .off = (s32)(p - file), .len = len };
		p += len + 1;
	}
	in->meta = meta;

	bench_input_end();
	if (!filter(in))
		return false;

	s32 sum = in->meta[in->num - 1].off + in->meta[in->num - 1].len + 1;
	float average_length = (float)sum / (float)in->num - 1.0f;
	pinfo("Loaded %d sequences", in->num);
	pinfol("Average sequence length: %.2f", average_length);
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
	.help = "Input file path: FASTA, DSV (.csv, .tsv, etc.)",
	.param = "FILE",
	.lopt = "input",
	.opt = 'i',
};
