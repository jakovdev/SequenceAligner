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

#define fline_next(fline, stream, line_len) \
	((line_len = getline(&fline->line, &fline->line_cap, stream)) != -1)

static bool empty(const char *line)
{
	return !*line || *line == '\n' || *line == '\r';
}

static long fpos_tell(FILE *stream)
{
	long pos = ftell(stream);
	if (pos < 0) {
		perr("Failed to read DSV file");
		exit(EXIT_FAILURE);
	}
	return pos;
}

static void fpos_seek(FILE *stream, long pos)
{
	if (fseek(stream, pos, SEEK_SET) != 0) {
		perr("Failed to parse DSV file");
		exit(EXIT_FAILURE);
	}
}

static bool extract_column_token(char *line, char delimiter, size_t target_col,
				 char **out_token, size_t *out_len)
{
	char delim_str[3];
	delim_str[0] = delimiter;
	delim_str[1] = '\n';
	delim_str[2] = '\0';

	char *token = strtok(line, delim_str);
	size_t current_col = 0;

	while (token && current_col < target_col) {
		current_col++;
		token = strtok(NULL, delim_str);
	}

	if (token && current_col == target_col) {
		size_t length = strlen(token);
		if (length > 0 && token[length - 1] == '\r')
			length--;
		*out_token = token;
		*out_len = length;
		return true;
	}

	return false;
}

bool dsv_detect(struct ifile *ifile, const char *restrict extension)
{
	if (!ifile || !extension || !*extension)
		return false;

	struct {
		const char *const ext;
		char delim;
	} const DSV_MAPPINGS[] = {
		{ "csv", ',' },	    { "tsv", '\t' }, { "tab", '\t' },
		{ "psv", '|' },	    { "ssv", ';' },  { "dsv", '\0' },
		{ "txt", '\0' },    { "dat", '\0' }, { "data", '\0' },
		{ "values", '\0' },
	};

	for (size_t i = 0; i < ARRAY_SIZE(DSV_MAPPINGS); i++) {
		if (strcasecmp(extension, DSV_MAPPINGS[i].ext) == 0) {
			ifile->format = INPUT_FORMAT_DSV;
			ifile->ctx.dsv.delimiter = DSV_MAPPINGS[i].delim;
			return true;
		}
	}

	return false;
}

static size_t dsv_count_columns(const char *line, char delimiter)
{
	if (!line || empty(line))
		return 0;

	size_t count = 1;
	while (!empty(line)) {
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

	long pos = fpos_tell(stream);
	long line_len;
	char detected = '\0';
	size_t best_score = 0;

	while (fline_next(ifile, stream, line_len)) {
		if (!line_len || empty(ifile->line))
			continue;
		break;
	}

	const char COMMON_DELIMITERS[] = { ',', '\t', ';', '|', ':' };
	if (line_len != -1) {
		for (size_t i = 0; i < ARRAY_SIZE(COMMON_DELIMITERS); i++) {
			char delim = COMMON_DELIMITERS[i];
			size_t count = dsv_count_columns(ifile->line, delim);

			/* Heuristic: prefer delimiters that give 2-100 columns */
			if (count >= 2 && count <= 100 && count > best_score) {
				best_score = count;
				detected = delim;
			}
		}
	}

	fpos_seek(stream, pos);

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

	long line_len;
	size_t line_number = 0;
	size_t expected_cols = 0;
	bool first_line = true;

	while (fline_next(ifile, stream, line_len)) {
		line_number++;

		if (!line_len || empty(ifile->line))
			continue;

		if (line_len > 1 &&
		    memchr(ifile->line, '\0', (size_t)(line_len - 1))) {
			perr("DSV file corruption on line %zu", line_number);
			rewind(stream);
			return false;
		}

		size_t curr_cols = dsv_count_columns(ifile->line, delimiter);

		if (first_line) {
			if (curr_cols == 0) {
				perr("DSV header is empty");
				rewind(stream);
				return false;
			}
			expected_cols = curr_cols;
			first_line = false;
		} else if (curr_cols > 0 && curr_cols != expected_cols) {
			perr("Expected %zu column(s), found %zu on line %zu",
			     expected_cols, curr_cols, line_number);
			rewind(stream);
			return false;
		}
	}

	if (first_line) {
		perr("DSV file is empty");
		rewind(stream);
		return false;
	}

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
	char delimiter = ifile->ctx.dsv.delimiter;

	rewind(stream);

	if (!delimiter && !dsv_delimiter(ifile))
		return false;

	if (!dsv_validate(ifile))
		return false;

	long line_len;

	if (!fline_next(ifile, stream, line_len)) {
		perr("Failed to read DSV header");
		return false;
	}

	size_t num_columns = dsv_count_columns(ifile->line, delimiter);
	pverb("Found %zu column(s) in DSV file", num_columns);

	if (!num_columns) {
		perr("Invalid DSV header");
		return false;
	}

	char **MALLOCA(headers, num_columns);
	if unlikely (!headers) {
		perr("Out of memory while reading DSV file");
		exit(EXIT_FAILURE);
	}

	for (size_t i = 0; i < num_columns; i++)
		headers[i] = NULL;

	char delim_str[2] = { delimiter, '\0' };
	char *token = strtok(ifile->line, delim_str);
	size_t column = 0;

	while (token && column < num_columns) {
		size_t token_len = strlen(token);
		while (token_len > 0 &&
		       (token[token_len - 1] == '\n' ||
			token[token_len - 1] == '\r' ||
			isspace((uchar)token[token_len - 1]))) {
			token[token_len - 1] = '\0';
			token_len--;
		}

		MALLOCA(headers[column], token_len + 1);
		if unlikely (!headers[column]) {
			perr("Out of memory while reading DSV file");
			for (size_t i = 0; i < column; i++) {
				if (headers[i])
					free(headers[i]);
			}
			free(headers);
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
		no_header = true;
		if (num_columns >= 2) {
			bench_io_end();
			pinfol("Select the column that contains sequence data:");
			char **MALLOCA(chs, num_columns + 1);
			if unlikely (!chs) {
				perr("Out of memory while reading DSV file");
				for (column = 0; column < num_columns; column++)
					free(headers[column]);
				free(headers);
				exit(EXIT_FAILURE);
			}

			for (column = 0; column < num_columns; column++)
				chs[column] = headers[column];

			size_t cn = num_columns;
			seq_col =
				(size_t)pchoice(chs, cn, "Enter column number");
			free(chs);
			bench_io_start();
		} else {
			pinfol("Only one column present; using it as the sequence column");
			seq_col = 0;
		}
	}

	if (seq_col < num_columns && headers && headers[seq_col])
		pverb("Using column #%zu ('%s') for sequences", seq_col + 1,
		      headers[seq_col]);

	for (column = 0; column < num_columns; column++)
		free(headers[column]);
	free(headers);

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

	long pos = fpos_tell(stream);
	long line_len;
	size_t count = 0;

	while (fline_next(ifile, stream, line_len)) {
		if (line_len && !empty(ifile->line))
			count++;
	}

	fpos_seek(stream, pos);

	return count;
}

void dsv_sequence_length(struct ifile *ifile, size_t *out_length)
{
	if unlikely (!ifile || !ifile->stream || !out_length) {
		pdev("Invalid parameters in dsv_sequence_length()");
		perr("Internal error getting sequence length in DSV file");
		pabort();
	}

	FILE *stream = ifile->stream;
	size_t column = ifile->ctx.dsv.sequence_column;
	char delimiter = ifile->ctx.dsv.delimiter;

	long pos = fpos_tell(stream);
	long line_len;
	size_t length = 0;

	if (!fline_next(ifile, stream, line_len)) {
		perr("Possible DSV file corruption, unexpected end of file");
		exit(EXIT_FAILURE);
	}

	char *token;
	if (!extract_column_token(ifile->line, delimiter, column, &token,
				  &length)) {
		perr("Possible DSV file corruption, could not extract sequence column");
		exit(EXIT_FAILURE);
	}

	fpos_seek(stream, pos);

	if (!length) {
		perr("Possible DSV file corruption, empty sequence found");
		exit(EXIT_FAILURE);
	}

	*out_length = length;
}

void dsv_sequence_extract(struct ifile *ifile, char *restrict output,
			  size_t expected_length)
{
	if unlikely (!ifile || !ifile->stream || !output) {
		pdev("Invalid parameters for dsv_sequence_extract()");
		perr("Internal error extracting sequence from DSV file");
		pabort();
	}

	FILE *stream = ifile->stream;
	char delimiter = ifile->ctx.dsv.delimiter;
	size_t column = ifile->ctx.dsv.sequence_column;

	long pos = fpos_tell(stream);
	long line_len;
	*output = '\0';

	if (!fline_next(ifile, stream, line_len)) {
		perr("Possible DSV file corruption, unexpected end of file");
		exit(EXIT_FAILURE);
	}

	char *token;
	size_t length;
	if (!extract_column_token(ifile->line, delimiter, column, &token,
				  &length)) {
		perr("Possible DSV file corruption, could not extract sequence column");
		exit(EXIT_FAILURE);
	}

	if (length != expected_length) {
		perr("Possible DSV file corruption, sequence length mismatch");
		exit(EXIT_FAILURE);
	}

	memcpy(output, token, length);
	output[length] = '\0';
	fpos_seek(stream, pos);
}

bool dsv_sequence_next(struct ifile *ifile)
{
	if unlikely (!ifile || !ifile->stream) {
		pdev("Invalid ifile in dsv_sequence_next()");
		perr("Internal error during DSV parsing");
		pabort();
	}

	FILE *stream = ifile->stream;
	char delimiter = ifile->ctx.dsv.delimiter;
	size_t expected_columns = ifile->ctx.dsv.num_columns;

	long line_len;

	while (fline_next(ifile, stream, line_len)) {
		if (!line_len || empty(ifile->line))
			continue;

		size_t cols = dsv_count_columns(ifile->line, delimiter);
		if (cols == expected_columns)
			break;

		perr("Possible DSV file corruption, expected %zu column(s), found %zu",
		     expected_columns, cols);
		exit(EXIT_FAILURE);
	}

	long pos = fpos_tell(stream);
	bool has_next = false;

	if (fline_next(ifile, stream, line_len)) {
		if (line_len && !empty(ifile->line))
			has_next = true;
	}

	fpos_seek(stream, pos);
	return has_next;
}
