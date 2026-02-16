#include "io/format/fasta.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "io/input.h"
#include "system/compiler.h"
#include "system/memory.h"
#include "util/benchmark.h"
#include "util/print.h"

#define fline_next(fline, stream, line_len) \
	((line_len = getline(&fline->line, &fline->line_cap, stream)) != -1)

static bool header(const char *line)
{
	return *line == '>';
}

#define for_each_nws(line, line_len, i)                                      \
	for (long i = 0; i < line_len && line[i] != '\n' && line[i] != '\r'; \
	     i++)                                                            \
		if (!isspace((uchar)line[i]))

static bool whitespace(const char *line, long line_len)
{
	for_each_nws(line, line_len, i)
		return false;
	return true;
}

static long fpos_tell(FILE *stream)
{
	long pos = ftell(stream);
	if (pos < 0) {
		perr("Failed to read FASTA file");
		exit(EXIT_FAILURE);
	}
	return pos;
}

static void fpos_seek(FILE *stream, long pos)
{
	if (fseek(stream, pos, SEEK_SET) != 0) {
		perr("Failed to parse FASTA file");
		exit(EXIT_FAILURE);
	}
}

bool fasta_detect(struct ifile *ifile, const char *restrict extension)
{
	if (!ifile || !extension || !*extension)
		return false;

	const char *const FASTA_EXTENSIONS[] = {
		"fasta", "fa", "fas", "fna", "ffn", "faa", "frn", "mpfa"
	};

	for (size_t i = 0; i < ARRAY_SIZE(FASTA_EXTENSIONS); i++) {
		if (strcasecmp(extension, FASTA_EXTENSIONS[i]) == 0) {
			ifile->format = INPUT_FORMAT_FASTA;
			return true;
		}
	}

	return false;
}

static bool fasta_validate(struct ifile *ifile)
{
	FILE *stream = ifile->stream;

	rewind(stream);

	long line_len;
	size_t seq_n = 0;
	size_t line_number = 0;
	bool in_sequence = false;
	bool found_sequence_data = false;
	bool ask_skip = false;
	bool skip = false;

	while (fline_next(ifile, stream, line_len)) {
		line_number++;

		if (line_len > 1 &&
		    memchr(ifile->line, '\0', (size_t)(line_len - 1))) {
			perr("FASTA file corruption on line %zu", line_number);
			rewind(stream);
			return false;
		}

		if (whitespace(ifile->line, line_len))
			continue;

		if (header(ifile->line)) {
			if (in_sequence && !found_sequence_data) {
				perr("Empty sequence found before line %zu",
				     line_number);
				rewind(stream);
				return false;
			}

			if (whitespace(ifile->line + 1, line_len - 1)) {
				if (!ask_skip) {
					bench_io_end();
					pwarn("Empty FASTA header found on line %zu",
					      line_number);
					skip = print_yN("Skip empty headers?");
					ask_skip = true;
					bench_io_start();
				}

				if (!skip) {
					perr("Empty FASTA header on line %zu",
					     line_number);
					rewind(stream);
					return false;
				}
			}

			seq_n++;
			in_sequence = true;
			found_sequence_data = false;
		} else if (in_sequence && !whitespace(ifile->line, line_len)) {
			found_sequence_data = true;
		} else {
			perr("Data found before first FASTA header on line %zu",
			     line_number);
			rewind(stream);
			return false;
		}
	}

	if (!seq_n) {
		perr("FASTA file contains no sequences");
		rewind(stream);
		return false;
	}

	if (in_sequence && !found_sequence_data) {
		perr("Last FASTA header has no sequence data");
		rewind(stream);
		return false;
	}

	rewind(stream);
	return true;
}

bool fasta_open(struct ifile *ifile)
{
	if unlikely (!ifile || !ifile->stream) {
		pdev("Invalid ifile in fasta_open()");
		perr("Internal error opening FASTA file");
		pabort();
	}

	if (!fasta_validate(ifile))
		return false;

	return true;
}

size_t fasta_sequence_count(struct ifile *ifile)
{
	if unlikely (!ifile || !ifile->stream) {
		pdev("Invalid ifile in fasta_sequence_count()");
		perr("Internal error counting FASTA sequences");
		pabort();
	}

	FILE *stream = ifile->stream;

	long pos = fpos_tell(stream);
	long line_len;
	size_t count = 0;

	while (fline_next(ifile, stream, line_len)) {
		if (line_len > 0 && header(ifile->line))
			count++;
	}

	fpos_seek(stream, pos);

	return count;
}

void fasta_sequence_length(struct ifile *ifile, size_t *out_length)
{
	if unlikely (!ifile || !ifile->stream || !out_length) {
		pdev("Invalid parameters in fasta_sequence_length()");
		perr("Internal error getting sequence length in FASTA file");
		pabort();
	}

	FILE *stream = ifile->stream;

	long pos = fpos_tell(stream);
	long line_len;
	size_t length = 0;

	if (!fline_next(ifile, stream, line_len) || !header(ifile->line)) {
		perr("Possible FASTA file corruption, no header found");
		exit(EXIT_FAILURE);
	}

	while (fline_next(ifile, stream, line_len) && !header(ifile->line)) {
		for_each_nws(ifile->line, line_len, i)
			length++;
	}

	fpos_seek(stream, pos);

	if (!length) {
		perr("Possible FASTA file corruption, empty sequence found");
		exit(EXIT_FAILURE);
	}

	*out_length = length;
}

void fasta_sequence_extract(struct ifile *ifile, char *restrict output,
			    size_t expected_length)
{
	if unlikely (!ifile || !ifile->stream || !output || !expected_length) {
		pdev("Invalid parameters for fasta_sequence_extract()");
		perr("Internal error extracting sequence from FASTA file");
		pabort();
	}

	FILE *stream = ifile->stream;

	long pos = fpos_tell(stream);
	long line_len;

	if (!fline_next(ifile, stream, line_len) || !header(ifile->line)) {
		perr("Possible FASTA file corruption, no header found");
		exit(EXIT_FAILURE);
	}

	*output = '\0';
	char *write_pos = output;
	size_t length = 0;

	while (fline_next(ifile, stream, line_len) && !header(ifile->line)) {
		for_each_nws(ifile->line, line_len, i) {
			*write_pos++ = ifile->line[i];
			length++;
		}
	}

	if (length != expected_length) {
		perr("Possible FASTA file corruption, sequence length mismatch");
		exit(EXIT_FAILURE);
	}

	*write_pos = '\0';
	fpos_seek(stream, pos);
}

bool fasta_sequence_next(struct ifile *ifile)
{
	if unlikely (!ifile || !ifile->stream) {
		pdev("Invalid ifile in fasta_sequence_next()");
		perr("Internal error during fasta parsing");
		pabort();
	}

	FILE *stream = ifile->stream;

	long line_len;

	while (fline_next(ifile, stream, line_len)) {
		if (whitespace(ifile->line, line_len))
			continue;

		if (header(ifile->line))
			break;

		perr("Possible FASTA file corruption, expected header");
		exit(EXIT_FAILURE);
	}

	long pos = fpos_tell(stream);
	bool has_next = false;
	if (fline_next(ifile, stream, line_len)) {
		if (line_len && !whitespace(ifile->line, line_len) &&
		    header(ifile->line)) {
			has_next = true;
		}
	}
	fpos_seek(stream, pos);

	return has_next;
}
