#include "io/input.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "io/format/dsv.h"
#include "io/format/fasta.h"
#include "system/compiler.h"
#include "system/memory.h"
#include "system/os.h"
#include "util/args.h"
#include "util/benchmark.h"
#include "util/print.h"

static void detect_file_format(struct ifile PS(ifile, 1),
			       const char PRS(path, 1))
{
	if (!*path)
		unreachable();

	const char *ext = strrchr(path, '.');
	if (!ext)
		return;

	ext++;

	static bool (*const DETECT[])(struct ifile S(1), const char RS(1)) = {
		fasta_detect,
		dsv_detect,
	};

	for (size_t i = 0; i < ARRAY_SIZE(DETECT); i++) {
		if (DETECT[i](ifile, ext))
			return;
	}
}

bool ifile_open(struct ifile PS(ifile, 1), const char PRS(path, 1))
{
	if unlikely (!*path) {
		pdev("Empty input path for ifile_open()");
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

	static bool (*const OPEN[])(struct ifile S(1)) = {
		[INPUT_FORMAT_FASTA] = fasta_open,
		[INPUT_FORMAT_DSV] = dsv_open,
	};
	if (!OPEN[ifile->format](ifile)) {
		fclose(ifile->stream);
		ifile->stream = NULL;
		return false;
	}

	static size_t (*const ENTRY_COUNT[])(struct ifile S(1)) = {
		[INPUT_FORMAT_FASTA] = fasta_entry_count,
		[INPUT_FORMAT_DSV] = dsv_entry_count,
	};
	ifile->entries = ENTRY_COUNT[ifile->format](ifile);

	bench_io_end();
	return true;
}

void ifile_close(struct ifile PS(ifile, 1))
{
	if (ifile->stream)
		fclose(ifile->stream);

	free(ifile->line);
	memset(ifile, 0, sizeof(*ifile));
}

size_t ifile_entry_length(struct ifile PS(ifile, 1))
{
	static size_t (*const ENTRY_LENGTH[])(struct ifile S(1)) = {
		[INPUT_FORMAT_FASTA] = fasta_entry_length,
		[INPUT_FORMAT_DSV] = dsv_entry_length,
	};
	return ENTRY_LENGTH[ifile->format](ifile);
}

size_t ifile_entry_extract(struct ifile PS(ifile, 1), char PRS(output, 1))
{
	static size_t (*const ENTRY_EXTRACT[])(struct ifile S(1),
					       char RS(1)) = {
		[INPUT_FORMAT_FASTA] = fasta_entry_extract,
		[INPUT_FORMAT_DSV] = dsv_entry_extract,
	};
	return ENTRY_EXTRACT[ifile->format](ifile, output);
}

bool ifile_entry_next(struct ifile PS(ifile, 1))
{
	static bool (*const ENTRY_NEXT[])(struct ifile S(1)) = {
		[INPUT_FORMAT_FASTA] = fasta_entry_next,
		[INPUT_FORMAT_DSV] = dsv_entry_next,
	};
	return ENTRY_NEXT[ifile->format](ifile);
}

static const char *input_path;

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
	.dest = &input_path,
	.parse_callback = parse_path,
	.validate_callback = validate_input_path,
	.validate_phase = ARG_CALLBACK_IF_SET,
	.validate_order = ARG_ORDER_FIRST,
	.action_callback = print_input_path,
	.action_phase = ARG_CALLBACK_ALWAYS,
	.action_order = ARG_ORDER_FIRST,
	.help_order = ARG_ORDER_FIRST,
};
