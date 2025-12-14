#include "io/format/dsv.h"

#include <ctype.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "io/input.h"
#include "system/compiler.h"
#include "system/memory.h"
#include "util/benchmark.h"
#include "util/print.h"

static const char COMMON_DELIMITERS[] = { ',', '\t', ';', '|', ':' };

bool dsv_detect(struct ifile *ifile, const char *restrict extension)
{
	if (!ifile || !extension || !*extension)
		return false;

	if (strcasecmp(extension, "csv") == 0) {
		ifile->format = INPUT_FORMAT_DSV;
		ifile->ctx.dsv.delimiter = ',';
		return true;
	}
	if (strcasecmp(extension, "tsv") == 0 ||
	    strcasecmp(extension, "tab") == 0) {
		ifile->format = INPUT_FORMAT_DSV;
		ifile->ctx.dsv.delimiter = '\t';
		return true;
	} else if (strcasecmp(extension, "psv") == 0) {
		ifile->format = INPUT_FORMAT_DSV;
		ifile->ctx.dsv.delimiter = '|';
		return true;
	} else if (strcasecmp(extension, "ssv") == 0) {
		ifile->format = INPUT_FORMAT_DSV;
		ifile->ctx.dsv.delimiter = ';';
		return true;
	} else if (strcasecmp(extension, "dsv") == 0 ||
		   strcasecmp(extension, "txt") == 0 ||
		   strcasecmp(extension, "dat") == 0 ||
		   strcasecmp(extension, "data") == 0 ||
		   strcasecmp(extension, "values") == 0) {
		ifile->format = INPUT_FORMAT_DSV;
		return true;
	}

	return false;
}

static size_t dsv_count_columns(const char *line, char delimiter)
{
	if (!line || !*line || *line == '\n' || *line == '\r')
		return 0;

	size_t count = 1;
	while (*line && *line != '\n' && *line != '\r') {
		if (*line == delimiter)
			count++;
		line++;
	}

	return count;
}

static bool dsv_delimiter(struct ifile *ifile)
{
	if unlikely (!ifile || !ifile->stream) {
		pdev("Invalid ifile in dsv_delimiter()");
		perr("Internal error detecting DSV delimiter");
		pabort();
	}

	FILE *stream = ifile->stream;
	char *line = NULL;
	size_t line_cap = 0;
	long line_len;
	char detected = '\0';
	size_t best_score = 0;

	long pos = ftell(stream);
	if (pos < 0)
		return false;

	while ((line_len = getline(&line, &line_cap, stream)) != -1) {
		if (line_len == 0 || *line == '\n' || *line == '\r')
			continue;
		break;
	}

	if (line_len != -1) {
		for (size_t i = 0; i < ARRAY_SIZE(COMMON_DELIMITERS); i++) {
			char delim = COMMON_DELIMITERS[i];
			size_t count = dsv_count_columns(line, delim);

			/* Heuristic: prefer delimiters that give 2-100 columns */
			if (count >= 2 && count <= 100 && count > best_score) {
				best_score = count;
				detected = delim;
			}
		}
	}

	free(line);

	if (fseek(stream, pos, SEEK_SET) != 0) {
		perr("Failed to parse DSV file");
		exit(EXIT_FAILURE);
	}

	if (detected == '\0') {
		bench_io_end();
		char prompt[2] = { 0 };
		pwarn("Could not automatically detect delimiter");
		pinput_s(prompt, "Enter delimiter character");
		detected = *prompt ? *prompt : ',';
		bench_io_start();
	}

	if (detected == '\0') {
		perr("Could not detect delimiter in DSV file");
		return false;
	}

	ifile->ctx.dsv.delimiter = detected;
	pverb("Detected delimiter: '%c'", detected);
	return true;
}

static void dsv_sequence_column(char **headers, size_t num_cols,
				size_t *seq_col)
{
	if (!headers || num_cols == 0 || num_cols > INT_MAX) {
		*seq_col = SIZE_MAX;
		return;
	}

	const char *seq_words[] = { "sequence", "seq",	 "protein", "dna",
				    "rna",	"amino", "peptide", "chain" };
	const size_t num_words = ARRAY_SIZE(seq_words);

	/* exact match */
	for (size_t column = 0; column < num_cols; column++) {
		for (size_t key = 0; key < num_words; key++) {
			if (strcasecmp(headers[column], seq_words[key]) == 0) {
				*seq_col = column;
				return;
			}
		}
	}

	/* partial match */
	for (size_t column = 0; column < num_cols; column++) {
		for (size_t key = 0; key < num_words; key++) {
			if (strcasestr(headers[column], seq_words[key])) {
				*seq_col = column;
				return;
			}
		}
	}

	/* no match */
	*seq_col = SIZE_MAX;
}

static bool dsv_validate(struct ifile *ifile)
{
	if unlikely (!ifile || !ifile->stream) {
		pdev("Invalid ifile in dsv_validate()");
		perr("Internal error validating DSV file");
		pabort();
	}

	FILE *stream = ifile->stream;
	char delimiter = ifile->ctx.dsv.delimiter;

	rewind(stream);
	char *line = NULL;
	size_t line_cap = 0;
	long line_len;

	size_t line_number = 0;
	size_t expected_columns = 0;
	bool first_line = true;

	while ((line_len = getline(&line, &line_cap, stream)) != -1) {
		line_number++;

		if (line_len == 0 || *line == '\n' || *line == '\r')
			continue;

		if (line_len > 1 &&
		    memchr(line, '\0', (size_t)(line_len - 1))) {
			perr("DSV file corruption on line %zu", line_number);
			free(line);
			rewind(stream);
			return false;
		}

		size_t current_columns = dsv_count_columns(line, delimiter);

		if (first_line) {
			if (current_columns == 0) {
				perr("DSV header is empty");
				free(line);
				rewind(stream);
				return false;
			}
			expected_columns = current_columns;
			first_line = false;
		} else if (current_columns > 0) {
			if (current_columns != expected_columns) {
				perr("Expected %zu column(s), found %zu on line %zu",
				     expected_columns, current_columns,
				     line_number);
				free(line);
				rewind(stream);
				return false;
			}
		}
	}

	if (first_line) {
		perr("DSV file is empty");
		free(line);
		rewind(stream);
		return false;
	}

	free(line);
	rewind(stream);
	return true;
}

bool dsv_open(struct ifile *ifile)
{
	if unlikely (!ifile || !ifile->stream) {
		pdev("Invalid ifile in dsv_open()");
		perr("Internal error opening DSV file");
		pabort();
	}

	FILE *stream = ifile->stream;
	rewind(stream);

	if (!ifile->ctx.dsv.delimiter && !dsv_delimiter(ifile))
		return false;

	if (!dsv_validate(ifile))
		return false;

	char *line = NULL;
	size_t line_cap = 0;
	long line_len;

	line_len = getline(&line, &line_cap, stream);
	if (line_len == -1) {
		perr("Failed to read DSV header");
		free(line);
		return false;
	}

	char delimiter = ifile->ctx.dsv.delimiter;
	size_t num_columns = dsv_count_columns(line, delimiter);
	pverb("Found %zu column(s) in DSV file", num_columns);

	if (!num_columns) {
		perr("Invalid DSV header");
		free(line);
		return false;
	}

	char **MALLOCA(headers, num_columns);
	if unlikely (!headers) {
		perr("Out of memory while reading DSV file");
		free(line);
		exit(EXIT_FAILURE);
	}

	for (size_t i = 0; i < num_columns; i++)
		headers[i] = NULL;

	char delim_str[2] = { delimiter, '\0' };
	char *token = strtok(line, delim_str);
	size_t column = 0;

	while (token && column < num_columns) {
		size_t token_len = strlen(token);
		while (token_len > 0 &&
		       (token[token_len - 1] == '\n' ||
			token[token_len - 1] == '\r' ||
			isspace((unsigned char)token[token_len - 1]))) {
			token[token_len - 1] = '\0';
			token_len--;
		}

		MALLOCA(headers[column], token_len + 1);
		if unlikely (!headers[column]) {
			perr("Out of memory while reading DSV file");
			for (size_t i = 0; i < column; i++)
				free(headers[i]);
			free(headers);
			free(line);
			exit(EXIT_FAILURE);
		}
		memcpy(headers[column], token, token_len + 1);
		column++;
		token = strtok(NULL, delim_str);
	}

	size_t seq_col = SIZE_MAX;
	dsv_sequence_column(headers, num_columns, &seq_col);

	if (seq_col == SIZE_MAX) {
		bench_io_end();
		char **MALLOCA(chs, num_columns + 2);
		if unlikely (!chs) {
			perr("Out of memory while reading DSV file");
			for (column = 0; column < num_columns; column++)
				free(headers[column]);
			free(headers);
			free(line);
			exit(EXIT_FAILURE);
		}

		for (column = 0; column < num_columns; column++)
			chs[column] = headers[column];

		char c_headerless[] = "My DSV file does not have a header line";
		chs[num_columns] = c_headerless;
		size_t cn = num_columns + 1;

		pinfo("Could not automatically detect the sequence column");
		pinfol("Which column contains your sequences?");
		seq_col = (size_t)pchoice(chs, cn, "Enter column number");
		free(chs);
		bench_io_start();
	}

	bool no_header = false;

	if (seq_col == num_columns) {
		if (num_columns >= 2) {
			bench_io_end();
			pinfol("Select the column that contains sequence data:");
			char **MALLOCA(chs, num_columns + 1);
			if unlikely (!chs) {
				perr("Out of memory while reading DSV file");
				for (column = 0; column < num_columns; column++)
					free(headers[column]);
				free(headers);
				free(line);
				exit(EXIT_FAILURE);
			}

			for (column = 0; column < num_columns; column++)
				chs[column] = headers[column];

			size_t cn = num_columns;
			seq_col =
				(size_t)pchoice(chs, cn, "Enter column number");
			no_header = true;
			bench_io_start();
		} else {
			pinfol("Only one column present; using it as the sequence column");
			seq_col = 0;
			no_header = true;
		}
	}

	if (seq_col < num_columns && headers && headers[seq_col])
		pverb("Using column #%zu ('%s') for sequences", seq_col + 1,
		      headers[seq_col]);

	for (column = 0; column < num_columns; column++)
		free(headers[column]);
	free(headers);
	free(line);

	ifile->ctx.dsv.sequence_column = seq_col;
	ifile->ctx.dsv.num_columns = num_columns;

	if (no_header)
		rewind(stream);

	return true;
}

size_t dsv_sequence_count(struct ifile *ifile)
{
	if unlikely (!ifile || !ifile->stream) {
		pdev("Invalid ifile in dsv_sequence_count()");
		perr("Internal error counting DSV lines");
		pabort();
	}

	FILE *stream = ifile->stream;
	long pos = ftell(stream);
	if (pos < 0) {
		perr("Failed to read DSV file");
		exit(EXIT_FAILURE);
	}

	char *line = NULL;
	size_t line_cap = 0;
	long line_len;
	size_t count = 0;

	while ((line_len = getline(&line, &line_cap, stream)) != -1) {
		if (line_len > 0 && *line != '\n' && *line != '\r' &&
		    *line != '\0')
			count++;
	}

	free(line);

	if (fseek(stream, pos, SEEK_SET) != 0) {
		perr("Failed to parse DSV file");
		exit(EXIT_FAILURE);
	}

	return count;
}

size_t dsv_sequence_length(struct ifile *ifile)
{
	if unlikely (!ifile || !ifile->stream) {
		pdev("Invalid ifile in dsv_sequence_length()");
		perr("Internal error getting sequence lengths in DSV file");
		pabort();
	}

	FILE *stream = ifile->stream;
	long pos = ftell(stream);
	if (pos < 0) {
		perr("Failed to read DSV file");
		exit(EXIT_FAILURE);
	}

	char *line = NULL;
	size_t line_cap = 0;
	long line_len;
	size_t column = ifile->ctx.dsv.sequence_column;
	char delimiter = ifile->ctx.dsv.delimiter;
	size_t length = 0;

	line_len = getline(&line, &line_cap, stream);
	if (line_len != -1) {
		char delim_str[3];
		delim_str[0] = delimiter;
		delim_str[1] = '\n';
		delim_str[2] = '\0';

		char *token = strtok(line, delim_str);
		size_t current_col = 0;

		while (token && current_col < column) {
			current_col++;
			token = strtok(NULL, delim_str);
		}

		if (token && current_col == column) {
			length = strlen(token);
			if (length > 0 && token[length - 1] == '\r')
				length--;
		}
	}

	free(line);

	if (fseek(stream, pos, SEEK_SET) != 0) {
		perr("Failed to parse DSV file");
		exit(EXIT_FAILURE);
	}

	return length;
}

size_t dsv_sequence_extract(struct ifile *ifile, char *restrict output)
{
	if unlikely (!ifile || !ifile->stream || !output) {
		pdev("Invalid parameters for dsv_sequence_extract()");
		perr("Internal error getting sequences from DSV file");
		pabort();
	}

	char *line = NULL;
	size_t line_cap = 0;
	long line_len;
	*output = '\0';
	size_t length = 0;
	size_t column = ifile->ctx.dsv.sequence_column;
	char delimiter = ifile->ctx.dsv.delimiter;
	FILE *stream = ifile->stream;

	line_len = getline(&line, &line_cap, stream);
	if (line_len != -1) {
		char delim_str[3];
		delim_str[0] = delimiter;
		delim_str[1] = '\n';
		delim_str[2] = '\0';

		char *token = strtok(line, delim_str);
		size_t current_col = 0;

		while (token && current_col < column) {
			current_col++;
			token = strtok(NULL, delim_str);
		}

		if (token && current_col == column) {
			length = strlen(token);
			if (length > 0 && token[length - 1] == '\r')
				length--;
			memcpy(output, token, length);
			output[length] = '\0';
		}
	}

	free(line);
	return length;
}

bool dsv_sequence_next(struct ifile *ifile)
{
	if unlikely (!ifile || !ifile->stream) {
		pdev("Invalid ifile in dsv_sequence_next()");
		perr("Internal error during DSV parsing");
		pabort();
	}

	FILE *stream = ifile->stream;
	long pos = ftell(stream);
	if (pos < 0)
		return false;

	char *line = NULL;
	size_t line_cap = 0;
	long line_len;
	char delimiter = ifile->ctx.dsv.delimiter;
	bool has_next = false;

	while ((line_len = getline(&line, &line_cap, stream)) != -1) {
		if (line_len == 0 || *line == '\n' || *line == '\r' ||
		    *line == '\0')
			continue;

		size_t cols = dsv_count_columns(line, delimiter);
		if (cols == ifile->ctx.dsv.num_columns) {
			has_next = true;
			break;
		}
	}

	free(line);

	if (fseek(stream, pos, SEEK_SET) != 0) {
		perr("Failed to parse DSV file");
		exit(EXIT_FAILURE);
	}

	return has_next;
}
