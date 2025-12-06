#include "io/format/fasta.h"

#include <stdlib.h>

#include "util/benchmark.h"
#include "util/print.h"

static char *fasta_skip_empty(char *cursor, char *file_end)
{
	while (cursor < file_end) {
		if (*cursor == ' ' || *cursor == '\t') {
			cursor++;
		} else if (*cursor == '\n' || *cursor == '\r') {
			if (*cursor == '\r' && cursor + 1 < file_end &&
			    *(cursor + 1) == '\n')
				cursor += 2;
			else
				cursor++;

		} else {
			break;
		}
	}

	return cursor;
}

static char *fasta_skip_line(char *cursor, char *file_end)
{
	while (cursor < file_end && !(*cursor == '\n' || *cursor == '\r'))
		cursor++;

	if (cursor >= file_end)
		return cursor;

	if (*cursor == '\r' && cursor + 1 < file_end && *(cursor + 1) == '\n')
		cursor += 2;
	else
		cursor++;
	return cursor;
}

bool fasta_validate(char *restrict file_start, char *restrict file_end)
{
	if (!file_start || !file_end || file_start >= file_end) {
		perr("Invalid file bounds for validation");
		return false;
	}

	char *cursor = file_start;
	size_t seq_n = 0;
	size_t line_number = 1;
	bool in_sequence = false;
	bool found_sequence_data = false;
	bool ask_skip = false;
	bool skip = false;

	cursor = fasta_skip_empty(cursor, file_end);
	while (cursor < file_end) {
		char ch = *cursor;

		if (ch == '\0') {
			perr("Null character found on line %zu", line_number);
			return false;
		}

		if (ch == '>') {
			if (in_sequence && !found_sequence_data) {
				perr("Empty sequence found on line %zu",
				     line_number);
				return false;
			}

			if (cursor + 1 < file_end &&
			    (*(cursor + 1) == '\n' || *(cursor + 1) == '\r')) {
				if (!ask_skip) {
					bench_io_end();
					pwarn("Empty header found on line %zu",
					      line_number);
					skip = print_yN("Skip empty headers?");
					ask_skip = true;
					bench_io_start();
				}

				if (!skip) {
					perr("Empty header found on line %zu",
					     line_number);
					return false;
				}
			}

			seq_n++;
			in_sequence = true;
			found_sequence_data = false;

			cursor = fasta_skip_line(cursor, file_end);
			line_number++;
		} else if (in_sequence) {
			if (ch == '\n' || ch == '\r') {
				line_number++;
				cursor = fasta_skip_empty(cursor, file_end);
			} else if (ch == ' ' || ch == '\t') {
				cursor++;
			} else {
				found_sequence_data = true;
				cursor++;
			}
		} else {
			perr("Data found before first header on line %zu",
			     line_number);
			return false;
		}
	}

	if (seq_n == 0) {
		perr("No sequences found in input file");
		return false;
	}

	if (in_sequence && !found_sequence_data) {
		perr("Last sequence header has no sequence data");
		return false;
	}

	return true;
}

size_t fasta_total_entries(char *restrict file_cursor, char *restrict file_end)
{
	size_t count = 0;
	char *cursor = file_cursor;

	while (cursor < file_end) {
		cursor = fasta_skip_empty(cursor, file_end);

		if (cursor >= file_end)
			break;

		if (*cursor == '>') {
			count++;
			cursor = fasta_skip_line(cursor, file_end);
		} else {
			cursor = fasta_skip_line(cursor, file_end);
		}
	}

	return count;
}

bool fasta_entry_next(char *restrict *restrict p_cursor)
{
	char *cursor = *p_cursor;
	while (*cursor) {
		if (*cursor == '>') {
			*p_cursor = cursor;
			return true;
		}

		cursor++;
	}

	*p_cursor = cursor;
	return false;
}

size_t fasta_entry_length(char *cursor, char *file_end)
{
	if (!cursor || !file_end || cursor >= file_end || *cursor != '>')
		return 0;

	cursor = fasta_skip_line(cursor, file_end);

	size_t length = 0;

	while (cursor < file_end) {
		cursor = fasta_skip_empty(cursor, file_end);

		if (cursor >= file_end || *cursor == '>')
			break;

		while (cursor < file_end &&
		       !(*cursor == '\n' || *cursor == '\r')) {
			if (*cursor != ' ' && *cursor != '\t')
				length++;

			cursor++;
		}

		if (cursor < file_end && (*cursor == '\n' || *cursor == '\r'))
			cursor = fasta_skip_line(cursor, file_end);
	}

	return length;
}

size_t fasta_entry_extract(char *restrict *restrict p_cursor,
			   char *restrict file_end, char *restrict output)
{
	if (!p_cursor || !output) {
		perr("Invalid parameters for fasta extraction");
		exit(EXIT_FAILURE);
	}

	char *cursor = *p_cursor;
	if (!cursor || cursor >= file_end || *cursor != '>') {
		*p_cursor = cursor;
		return 0;
	}

	cursor = fasta_skip_line(cursor, file_end);
	char *write_pos = output;
	size_t length = 0;

	while (cursor < file_end) {
		cursor = fasta_skip_empty(cursor, file_end);

		if (cursor >= file_end || *cursor == '>')
			break;

		while (cursor < file_end &&
		       !(*cursor == '\n' || *cursor == '\r')) {
			if (*cursor != ' ' && *cursor != '\t') {
				*write_pos++ = *cursor;
				length++;
			}

			cursor++;
		}

		if (cursor < file_end && (*cursor == '\n' || *cursor == '\r'))
			cursor = fasta_skip_line(cursor, file_end);
	}

	*write_pos = '\0';
	*p_cursor = cursor;
	return length;
}
