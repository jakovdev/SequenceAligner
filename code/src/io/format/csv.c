#include "io/format/csv.h"

#include <limits.h>
#include <stdint.h>
#include <string.h>

#include "system/memory.h"
#include "util/benchmark.h"
#include "util/print.h"

#ifdef _WIN32
#include "system/os.h"
#endif

static size_t csv_column_count(const char *line)
{
	if (!line || !*line || *line == '\n' || *line == '\r')
		return 0;

	size_t count = 1;
	while (*line && *line != '\n' && *line != '\r') {
		if (*line == ',')
			count++;
		line++;
	}

	return count;
}

static char *csv_column_copy(const char *restrict file_start,
			     const char *restrict file_end)
{
	ptrdiff_t delta = file_end - file_start;
	size_t len = delta < 0 ? 0 : (size_t)delta;
	char *MALLOCA(name, len + 1);
	if likely (name) {
		memcpy(name, file_start, len);
		name[len] = '\0';
	}

	return name;
}

static void csv_column_sequence(char **headers, size_t num_cols,
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

	/* No match found */
	*seq_col = SIZE_MAX;
}

char *csv_header_parse(char *restrict file_cursor, char *restrict file_end,
		       bool *no_header, size_t *seq_col)
{
	if unlikely (!file_cursor || !file_end || file_cursor >= file_end ||
		     !no_header || !seq_col) {
		pdev("Invalid parameters for csv header parsing");
		perr("Internal error during csv header processing, possible file corruption");
		exit(EXIT_FAILURE);
	}

	char *header_start = file_cursor;
	*seq_col = SIZE_MAX;
	*no_header = false;
	size_t num_columns = 0;
	char **headers = NULL;

	num_columns = csv_column_count(header_start);
	pverb("Found %zu column(s) in input file", num_columns);
	if (!num_columns) {
		perr("Invalid csv header (do you have an empty first line or file?)");
		exit(EXIT_FAILURE);
	}

	MALLOCA(headers, num_columns);
	if unlikely (!headers) {
		perr("Out of memory during csv header processing");
		exit(EXIT_FAILURE);
	}

	for (size_t i = 0; i < num_columns; i++)
		headers[i] = NULL;

	const char *col_start = header_start;
	size_t column = 0;

	while (file_cursor < file_end) {
		if (*file_cursor == ',' || *file_cursor == '\n' ||
		    *file_cursor == '\r') {
			if (column < num_columns) {
				headers[column] =
					csv_column_copy(col_start, file_cursor);
				column++;
			}

			if (*file_cursor == ',' && column < num_columns) {
				col_start = file_cursor + 1;
			} else if (*file_cursor == '\n') {
				file_cursor++;
				break;
			} else if (*file_cursor == '\r') {
				file_cursor++;
				if (file_cursor < file_end &&
				    *file_cursor == '\n')
					file_cursor++;

				break;
			}
		}

		file_cursor++;
	}

	csv_column_sequence(headers, num_columns, seq_col);
	if (*seq_col == SIZE_MAX) {
		bench_io_end();
		char **MALLOCA(chs, num_columns + 2);
		if unlikely (!chs) {
			perr("Out of memory during csv header processing");
			for (column = 0; column < num_columns; column++) {
				if (headers[column])
					free(headers[column]);
			}
			free(headers);
			exit(EXIT_FAILURE);
		}

		for (column = 0; column < num_columns; column++)
			chs[column] = headers[column];

		char c_headerless[] = "My csv file does not have a header line";
		chs[num_columns] = c_headerless;
		size_t cn = num_columns + 1;

		pinfo("Could not automatically detect the sequence column");
		pinfol("Which column contains your sequences?");
		*seq_col = (size_t)pchoice(chs, cn, "Enter column number");
		free(chs);
		bench_io_start();
	}

	if (*seq_col == num_columns) {
		if (num_columns >= 2) {
			bench_io_end();
			pinfol("OK, select the column that displays a sequence");
			char **chs = headers;
			size_t cn = num_columns;
			*seq_col =
				(size_t)pchoice(chs, cn, "Enter column number");
			*no_header = true;
			bench_io_start();
		} else {
			pinfol("Only one column present; using it as the sequence column");
			*seq_col = 0;
			*no_header = true;
		}
	}

	if (*seq_col < num_columns && headers && headers[*seq_col])
		pverb("Using column #%zu ('%s')", *seq_col + 1,
		      headers[*seq_col]);

	for (column = 0; column < num_columns; column++) {
		if (headers[column])
			free(headers[column]);
	}

	free(headers);
	headers = NULL;
	return file_cursor;
}

bool csv_line_next(char *restrict *restrict p_cursor)
{
	if unlikely (!p_cursor || !*p_cursor) {
		pdev("Invalid parameters for csv line iteration");
		perr("Internal error during csv parsing, possible file corruption");
		exit(EXIT_FAILURE);
	}

	char *cursor = *p_cursor;

	while (*cursor &&
	       (*cursor == ' ' || *cursor == '\r' || *cursor == '\n'))
		cursor++;

	if (!*cursor) {
		*p_cursor = cursor;
		return false;
	}

	while (*cursor && *cursor != '\n' && *cursor != '\r')
		cursor++;

	while (*cursor && (*cursor == '\n' || *cursor == '\r'))
		cursor++;

	*p_cursor = cursor;
	return true;
}

size_t csv_total_lines(char *restrict file_cursor, char *restrict file_end)
{
	if unlikely (!file_cursor || !file_end || file_cursor >= file_end) {
		pdev("Invalid file bounds for csv line counting");
		perr("Internal error counting csv lines, possible file corruption");
		exit(EXIT_FAILURE);
	}

	size_t total_lines = 0;

	while (file_cursor < file_end && *file_cursor) {
		if (csv_line_next(&file_cursor))
			total_lines++;
	}

	return total_lines;
}

size_t csv_line_column_extract(char *restrict *restrict p_cursor,
			       char *restrict output, size_t target_column)
{
	if unlikely (!p_cursor || !output) {
		pdev("Invalid parameters for csv column extraction");
		perr("Internal error during csv extraction, possible file corruption");
		exit(EXIT_FAILURE);
	}

	char *cursor = *p_cursor;
	char *write_pos = NULL;
	size_t column = 0;
	size_t column_length = 0;

	while (*cursor &&
	       (*cursor == ' ' || *cursor == '\r' || *cursor == '\n'))
		cursor++;

	while (*cursor && *cursor != '\n' && *cursor != '\r') {
		if (column == target_column) {
			write_pos = output;

			while (*cursor && *cursor != ',' && *cursor != '\n' &&
			       *cursor != '\r')
				*write_pos++ = *cursor++;

			*write_pos = '\0';
			ptrdiff_t delta = write_pos - output;
			column_length = delta < 0 ? 0 : (size_t)delta;
		} else {
			while (*cursor && *cursor != ',' && *cursor != '\n' &&
			       *cursor != '\r')
				cursor++;
		}

		if (*cursor == ',') {
			cursor++;
			column++;
		}
	}

	while (*cursor && (*cursor == '\n' || *cursor == '\r'))
		cursor++;

	*p_cursor = cursor;
	return column_length;
}

size_t csv_line_column_length(char *cursor, size_t target_column)
{
	if unlikely (!cursor) {
		pdev("Invalid parameters for csv column length calculation");
		perr("Internal error during csv length calculation, possible file corruption");
		exit(EXIT_FAILURE);
	}

	size_t column = 0;
	size_t column_length = 0;

	while (*cursor &&
	       (*cursor == ' ' || *cursor == '\r' || *cursor == '\n'))
		cursor++;

	while (*cursor && *cursor != '\n' && *cursor != '\r') {
		if (column == target_column) {
			char *col_start = cursor;
			while (*cursor && *cursor != ',' && *cursor != '\n' &&
			       *cursor != '\r')
				cursor++;

			ptrdiff_t delta = cursor - col_start;
			column_length = delta < 0 ? 0 : (size_t)delta;
			break;
		} else {
			while (*cursor && *cursor != ',' && *cursor != '\n' &&
			       *cursor != '\r')
				cursor++;
		}

		if (*cursor == ',') {
			cursor++;
			column++;
		}
	}

	return column_length;
}

bool csv_validate(const char *restrict file_start,
		  const char *restrict file_end)
{
	if unlikely (!file_start || !file_end || file_start >= file_end) {
		pdev("Invalid file bounds for csv validation");
		perr("Internal error validating csv, possible file corruption");
		exit(EXIT_FAILURE);
	}

	const char *cursor = file_start;
	size_t line_number = 0;
	size_t expected_columns = 0;
	bool first_line = true;

	while (cursor < file_end) {
		const char *line_start = cursor;
		while (cursor < file_end &&
		       (*cursor == ' ' || *cursor == '\t' || *cursor == '\r'))
			cursor++;

		if (cursor < file_end && *cursor == '\n') {
			cursor++;
			line_number++;
			continue;
		}

		if (cursor >= file_end)
			break;

		size_t current_columns = 0;
		bool field_started = false;

		while (cursor < file_end) {
			char ch = *cursor;

			if (ch == ',' || ch == '\n' || ch == '\r') {
				current_columns++;
				field_started = false;

				if (ch == '\n' || ch == '\r') {
					if (ch == '\r' &&
					    (cursor + 1 < file_end) &&
					    cursor[1] == '\n')
						cursor++;

					cursor++;
					line_number++;
					break;
				}
			} else if (ch == '\0') {
				perr("Null character found on line %zu",
				     line_number);
				return false;
			} else {
				field_started = true;
			}

			cursor++;
		}

		if (field_started ||
		    (cursor > line_start && cursor > file_start &&
		     *(cursor - 1) == ',')) {
			current_columns++;
		}

		if (first_line) {
			if (current_columns == 0) {
				perr("Header line has zero columns");
				return false;
			}

			expected_columns = current_columns;
			first_line = false;
		} else if (current_columns > 0) {
			if (current_columns != expected_columns) {
				perr("Expected %zu column(s), found %zu on line %zu",
				     expected_columns, current_columns,
				     line_number);
				return false;
			}
		}
	}

	return true;
}
