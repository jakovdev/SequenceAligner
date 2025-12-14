#include "io/format/fasta.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "io/input.h"
#include "system/compiler.h"
#include "util/benchmark.h"
#include "util/print.h"

static bool fasta_skip_empty_lines(FILE *stream)
{
	long pos;
	char *line = NULL;
	size_t line_cap = 0;
	long line_len;
	bool found = false;

	while ((line_len = getline(&line, &line_cap, stream)) != -1) {
		pos = ftell(stream);
		if (pos < 0) {
			free(line);
			return false;
		}

		bool is_empty = true;
		for (long i = 0; i < line_len; i++) {
			if (!isspace((unsigned char)line[i])) {
				is_empty = false;
				break;
			}
		}

		if (!is_empty) {
			if (fseek(stream, pos - line_len, SEEK_SET) != 0) {
				perr("Failed to parse FASTA file");
				exit(EXIT_FAILURE);
			}

			found = true;
			break;
		}
	}

	free(line);
	return found;
}

bool fasta_detect(struct ifile *ifile, const char *restrict extension)
{
	if (!ifile || !extension || !*extension)
		return false;

	if (strcasecmp(extension, "fasta") == 0 ||
	    strcasecmp(extension, "fa") == 0 ||
	    strcasecmp(extension, "fas") == 0 ||
	    strcasecmp(extension, "fna") == 0 ||
	    strcasecmp(extension, "ffn") == 0 ||
	    strcasecmp(extension, "faa") == 0 ||
	    strcasecmp(extension, "frn") == 0 ||
	    strcasecmp(extension, "mpfa") == 0) {
		ifile->format = INPUT_FORMAT_FASTA;
		return true;
	}

	return false;
}

static bool fasta_validate(struct ifile *ifile)
{
	FILE *stream = ifile->stream;
	rewind(stream);

	char *line = NULL;
	size_t line_cap = 0;
	long line_len;

	size_t seq_n = 0;
	size_t line_number = 0;
	bool in_sequence = false;
	bool found_sequence_data = false;
	bool ask_skip = false;
	bool skip = false;

	while ((line_len = getline(&line, &line_cap, stream)) != -1) {
		line_number++;

		if (line_len > 1 &&
		    memchr(line, '\0', (size_t)(line_len - 1))) {
			perr("FASTA file appears corrupted (null byte on line %zu)",
			     line_number);
			free(line);
			rewind(stream);
			return false;
		}

		bool is_empty = true;
		for (long i = 0;
		     i < line_len && line[i] != '\n' && line[i] != '\r'; i++) {
			if (!isspace((unsigned char)line[i])) {
				is_empty = false;
				break;
			}
		}

		if (is_empty)
			continue;

		if (*line == '>') {
			if (in_sequence && !found_sequence_data) {
				perr("Empty sequence found before line %zu",
				     line_number);
				free(line);
				rewind(stream);
				return false;
			}

			bool header_empty = true;
			for (long i = 1;
			     i < line_len && line[i] != '\n' && line[i] != '\r';
			     i++) {
				if (!isspace((unsigned char)line[i])) {
					header_empty = false;
					break;
				}
			}

			if (header_empty) {
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
					free(line);
					rewind(stream);
					return false;
				}
			}

			seq_n++;
			in_sequence = true;
			found_sequence_data = false;
		} else if (in_sequence) {
			for (long i = 0;
			     i < line_len && line[i] != '\n' && line[i] != '\r';
			     i++) {
				if (!isspace((unsigned char)line[i])) {
					found_sequence_data = true;
					break;
				}
			}
		} else {
			perr("Data found before first FASTA header on line %zu",
			     line_number);
			free(line);
			rewind(stream);
			return false;
		}
	}

	if (!seq_n) {
		perr("FASTA file contains no sequences");
		free(line);
		rewind(stream);
		return false;
	}

	if (in_sequence && !found_sequence_data) {
		perr("Last FASTA header has no sequence data");
		free(line);
		rewind(stream);
		return false;
	}

	free(line);
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
	long pos = ftell(stream);
	if (pos < 0) {
		perr("Failed to read FASTA file");
		exit(EXIT_FAILURE);
	}

	char *line = NULL;
	size_t line_cap = 0;
	long line_len;
	size_t count = 0;

	while ((line_len = getline(&line, &line_cap, stream)) != -1) {
		if (line_len > 0 && *line == '>')
			count++;
	}

	free(line);

	if (fseek(stream, pos, SEEK_SET) != 0) {
		perr("Failed to parse FASTA file");
		exit(EXIT_FAILURE);
	}

	return count;
}

size_t fasta_sequence_length(struct ifile *ifile)
{
	if unlikely (!ifile || !ifile->stream) {
		pdev("Invalid ifile in fasta_sequence_length()");
		perr("Internal error getting sequence lengthts in FASTA file");
		pabort();
	}

	FILE *stream = ifile->stream;
	long pos = ftell(stream);
	if (pos < 0) {
		perr("Failed to read FASTA file");
		exit(EXIT_FAILURE);
	}

	char *line = NULL;
	size_t line_cap = 0;
	long line_len;

	line_len = getline(&line, &line_cap, stream);
	if (line_len == -1 || *line != '>') {
		free(line);

		if (fseek(stream, pos, SEEK_SET) != 0) {
			perr("Failed to parse FASTA file");
			exit(EXIT_FAILURE);
		}

		return 0;
	}

	size_t length = 0;
	while ((line_len = getline(&line, &line_cap, stream)) != -1) {
		if (*line == '>')
			break;

		for (long i = 0;
		     i < line_len && line[i] != '\n' && line[i] != '\r'; i++) {
			if (!isspace((unsigned char)line[i]))
				length++;
		}
	}

	free(line);

	if (fseek(stream, pos, SEEK_SET) != 0) {
		perr("Failed to parse FASTA file");
		exit(EXIT_FAILURE);
	}

	return length;
}

size_t fasta_sequence_extract(struct ifile *ifile, char *restrict output)
{
	if unlikely (!ifile || !ifile->stream || !output) {
		pdev("Invalid parameters for fasta_sequence_extract()");
		perr("Internal error getting sequences from FASTA file");
		pabort();
	}

	FILE *stream = ifile->stream;
	char *line = NULL;
	size_t line_cap = 0;
	long line_len;

	line_len = getline(&line, &line_cap, stream);
	if (line_len == -1 || *line != '>') {
		free(line);
		return 0;
	}

	*output = '\0';
	char *write_pos = output;
	size_t length = 0;

	while ((line_len = getline(&line, &line_cap, stream)) != -1) {
		if (*line == '>') {
			long current = ftell(stream);
			if (current >= 0) {
				long line_start = current - line_len;
				if (fseek(stream, line_start, SEEK_SET) != 0) {
					perr("Failed to parse FASTA file");
					exit(EXIT_FAILURE);
				}
			} else {
				perr("Failed to parse FASTA file");
				exit(EXIT_FAILURE);
			}
			break;
		}

		for (long i = 0;
		     i < line_len && line[i] != '\n' && line[i] != '\r'; i++) {
			if (!isspace((unsigned char)line[i])) {
				*write_pos++ = line[i];
				length++;
			}
		}
	}

	*write_pos = '\0';
	free(line);
	return length;
}

bool fasta_sequence_next(struct ifile *ifile)
{
	if unlikely (!ifile || !ifile->stream) {
		pdev("Invalid ifile in fasta_sequence_next()");
		perr("Internal error during fasta parsing");
		pabort();
	}

	FILE *stream = ifile->stream;
	if (!fasta_skip_empty_lines(stream))
		return false;

	int c = fgetc(stream);
	if (c == EOF)
		return false;

	ungetc(c, stream);
	return c == '>';
}
