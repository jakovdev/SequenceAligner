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

static const struct {
	bool (*detect)(struct ifile[static 1], const char[restrict static 1]);
	enum input_format format;
} format_detectors[] = {
	{ fasta_detect, INPUT_FORMAT_FASTA },
	{ dsv_detect, INPUT_FORMAT_DSV },
};

static void detect_file_format(struct ifile ifile[static 1],
			       const char path[restrict static 1])
{
	if (!*path)
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

bool ifile_open(struct ifile ifile[static 1],
		const char path[restrict static 1])
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

	switch (ifile->format) {
	case INPUT_FORMAT_FASTA:
		ifile->entries = fasta_entry_count(ifile);
		break;
	case INPUT_FORMAT_DSV:
		ifile->entries = dsv_entry_count(ifile);
		break;
	case INPUT_FORMAT_UNKNOWN:
	default:
		unreachable();
	}

	bench_io_end();
	return true;
}

void ifile_close(struct ifile ifile[static 1])
{
	if (ifile->stream)
		fclose(ifile->stream);

	free(ifile->line);
	memset(ifile, 0, sizeof(*ifile));
}

void ifile_entry_length(struct ifile ifile[static 1],
			size_t out_length[restrict static 1])
{
	switch (ifile->format) {
	case INPUT_FORMAT_FASTA:
		fasta_entry_length(ifile, out_length);
		return;
	case INPUT_FORMAT_DSV:
		dsv_entry_length(ifile, out_length);
		return;
	case INPUT_FORMAT_UNKNOWN:
	default:
		pdev("Unknown format in ifile_entry_length()");
		perr("Internal error retrieving sequence length from file");
		pabort();
	}
}

void ifile_entry_extract(struct ifile ifile[static 1],
			 char output[restrict static 1], size_t expected_length)
{
	if (!expected_length) {
		pdev("Invalid length in ifile_entry_extract()");
		perr("Internal error retrieving sequence from file");
		pabort();
	}

	switch (ifile->format) {
	case INPUT_FORMAT_FASTA:
		fasta_entry_extract(ifile, output, expected_length);
		return;
	case INPUT_FORMAT_DSV:
		dsv_entry_extract(ifile, output, expected_length);
		return;
	case INPUT_FORMAT_UNKNOWN:
	default:
		pdev("Unknown format in ifile_entry_extract()");
		perr("Internal error retrieving sequence from file");
		pabort();
	}
}

bool ifile_entry_next(struct ifile ifile[static 1])
{
	switch (ifile->format) {
	case INPUT_FORMAT_FASTA:
		return fasta_entry_next(ifile);
	case INPUT_FORMAT_DSV:
		return dsv_entry_next(ifile);
	case INPUT_FORMAT_UNKNOWN:
	default:
		pdev("Unknown format in ifile_entry_next()");
		perr("Internal error during file parsing");
		pabort();
	}
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
