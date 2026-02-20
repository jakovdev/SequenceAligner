#include "io/input.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bio/types.h"
#include "io/format/dsv.h"
#include "io/format/fasta.h"
#include "system/compiler.h"
#include "system/memory.h"
#include "system/os.h"
#include "util/args.h"
#include "util/benchmark.h"
#include "util/print.h"

static const struct {
	bool (*detect)(struct ifile *, const char *restrict);
	enum input_format format;
} format_detectors[] = {
	{ fasta_detect, INPUT_FORMAT_FASTA },
	{ dsv_detect, INPUT_FORMAT_DSV },
};

static void detect_file_format(struct ifile *ifile, const char *path)
{
	if (!path || !*path)
		unreachable();

	const char *ext = strrchr(path, '.');
	if (!ext)
		return;

	ext++;

	for (size_t i = 0; i < ARRAY_SIZE(format_detectors); i++) {
		if (format_detectors[i].detect(ifile, ext))
			return;
	}

	return;
}

bool ifile_open(struct ifile *ifile, const char *restrict path)
{
	if unlikely (!ifile || !path || !*path) {
		pdev("NULL parameters for ifile_open()");
		perr("Internal error opening input file");
		pabort();
	}

	bench_io_start();
	memset(ifile, 0, sizeof(*ifile));
	detect_file_format(ifile, path);
	if (ifile->format == INPUT_FORMAT_UNKNOWN) {
		perr("Unsupported or unknown file format: %s", file_name(path));
		perr("Supported formats: FASTA, DSV (CSV, TSV, etc.)");
		return false;
	}

	ifile->stream = fopen(path, "r");
	if (!ifile->stream) {
		perr("Could not open file: %s", file_name(path));
		perr("Check that the file exists and has read permissions");
		return false;
	}

	bool opened = false;
	switch (ifile->format) {
	case INPUT_FORMAT_FASTA:
		opened = fasta_open(ifile);
		break;
	case INPUT_FORMAT_DSV:
		opened = dsv_open(ifile);
		break;
	case INPUT_FORMAT_UNKNOWN:
	default:
		unreachable();
	}

	if (!opened) {
		fclose(ifile->stream);
		ifile->stream = NULL;
		return false;
	}

	pverb("Counting sequences in input file");
	size_t total = 0;
	switch (ifile->format) {
	case INPUT_FORMAT_FASTA:
		total = fasta_sequence_count(ifile);
		break;
	case INPUT_FORMAT_DSV:
		total = dsv_sequence_count(ifile);
		break;
	case INPUT_FORMAT_UNKNOWN:
	default:
		unreachable();
	}

	if (total >= SEQ_N_MAX) {
		perr("Too many sequences in input file: %zu > max:%u", total,
		     SEQ_N_MAX);
		ifile_close(ifile);
		return false;
	}

	if (total < SEQ_N_MIN) {
		perr("Not enough sequences in input file: %zu < min:%u)", total,
		     SEQ_N_MIN);
		ifile_close(ifile);
		return false;
	}

	ifile->total_sequences = (s32)total;
	pinfo("Found " Ps32 " sequences", ifile->total_sequences);
	bench_io_end();
	return true;
}

void ifile_close(struct ifile *ifile)
{
	if unlikely (!ifile) {
		pdev("NULL ifile in ifile_close()");
		perr("Internal error while closing input file");
		pabort();
	}

	if (ifile->stream)
		fclose(ifile->stream);

	if (ifile->line)
		free(ifile->line);

	memset(ifile, 0, sizeof(*ifile));
}

s32 ifile_sequence_count(struct ifile *ifile)
{
	if unlikely (!ifile) {
		pdev("NULL ifile in ifile_sequence_count()");
		perr("Internal error retrieving total sequences from input file");
		pabort();
	}

	return ifile->total_sequences;
}

void ifile_sequence_length(struct ifile *ifile, size_t *out_length)
{
	if (ifile && out_length) {
		switch (ifile->format) {
		case INPUT_FORMAT_FASTA:
			fasta_sequence_length(ifile, out_length);
			return;
		case INPUT_FORMAT_DSV:
			dsv_sequence_length(ifile, out_length);
			return;
		case INPUT_FORMAT_UNKNOWN:
		default:
			break;
		}
	}

	pdev("NULL ifile or unknown format in ifile_sequence_length()");
	perr("Internal error retrieving sequence length from file");
	pabort();
}

void ifile_sequence_extract(struct ifile *ifile, char *restrict output,
			    size_t expected_length)
{
	if (ifile && output && expected_length) {
		switch (ifile->format) {
		case INPUT_FORMAT_FASTA:
			fasta_sequence_extract(ifile, output, expected_length);
			return;
		case INPUT_FORMAT_DSV:
			dsv_sequence_extract(ifile, output, expected_length);
			return;
		case INPUT_FORMAT_UNKNOWN:
		default:
			break;
		}
	}

	pdev("NULL file or unknown format in ifile_sequence_extract()");
	perr("Internal error retrieving sequence from file");
	pabort();
}

bool ifile_sequence_next(struct ifile *ifile)
{
	if (ifile) {
		switch (ifile->format) {
		case INPUT_FORMAT_FASTA:
			return fasta_sequence_next(ifile);
		case INPUT_FORMAT_DSV:
			return dsv_sequence_next(ifile);
		case INPUT_FORMAT_UNKNOWN:
		default:
			break;
		}
	}

	pdev("NULL file or unknown format in ifile_sequence_next()");
	perr("Internal error during file parsing");
	pabort();
}

static char input_path[MAX_PATH];

const char *arg_input(void)
{
	return input_path;
}

static void print_input_path(void)
{
	pinfo("Input: %s", file_name(input_path));
}

static struct arg_callback validate_input_path(void)
{
	if (!path_file_exists(input_path))
		return ARG_INVALID("Input file does not exist");

	return ARG_VALID();
}

ARGUMENT(input_path) = {
	.opt = 'i',
	.lopt = "input",
	.help = "Input file path: FASTA, DSV (CSV, TSV, etc.) format",
	.param = "FILE",
	.param_req = ARG_PARAM_REQUIRED,
	.arg_req = ARG_REQUIRED,
	.dest = input_path,
	.parse_callback = parse_path,
	.validate_callback = validate_input_path,
	.validate_phase = ARG_CALLBACK_IF_SET,
	.validate_order = ARG_ORDER_FIRST,
	.action_callback = print_input_path,
	.action_phase = ARG_CALLBACK_ALWAYS,
	.action_order = ARG_ORDER_FIRST,
	.help_order = ARG_ORDER_FIRST,
};
