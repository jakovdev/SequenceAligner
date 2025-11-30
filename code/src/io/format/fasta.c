#include "io/format/fasta.h"

#include "util/benchmark.h"
#include "util/print.h"

static inline char *fasta_skip_empty(char *cursor, char *file_end)
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

static inline char *fasta_skip_line(char *cursor, char *file_end)
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
		perror("Invalid file bounds for validation");
		return false;
	}

	char *cursor = file_start;
	u64 sequence_count = 0;
	u64 line_number = 1;
	bool in_sequence = false;
	bool found_sequence_data = false;
	bool ask_skip = false;
	bool skip = false;

	cursor = fasta_skip_empty(cursor, file_end);
	while (cursor < file_end) {
		char ch = *cursor;

		if (ch == '\0') {
			perror("Null character found on line " Pu64,
			       line_number);
			return false;
		}

		if (ch == '>') {
			if (in_sequence && !found_sequence_data) {
				perror("Empty sequence found on line " Pu64,
				       line_number);
				return false;
			}

			if (cursor + 1 < file_end &&
			    (*(cursor + 1) == '\n' || *(cursor + 1) == '\r')) {
				if (!ask_skip) {
					bench_io_end();
					pwarn("Empty header found on line " Pu64,
					      line_number);
					skip = print_yN("Skip empty headers?");
					ask_skip = true;
					bench_io_start();
				}

				if (!skip) {
					perror("Empty header found on line " Pu64,
					       line_number);
					return false;
				}
			}

			sequence_count++;
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
			perror("Data found before first header on line " Pu64,
			       line_number);
			return false;
		}
	}

	if (sequence_count == 0) {
		perror("No sequences found in input file");
		return false;
	}

	if (in_sequence && !found_sequence_data) {
		perror("Last sequence header has no sequence data");
		return false;
	}

	return true;
}

u64 fasta_total_entries(char *restrict file_cursor, char *restrict file_end)
{
	u64 count = 0;
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

u64 fasta_entry_length(char *cursor, char *file_end)
{
	if (!cursor || !file_end || cursor >= file_end || *cursor != '>')
		return 0;

	cursor = fasta_skip_line(cursor, file_end);

	u64 length = 0;

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

u64 fasta_entry_extract(char *restrict *restrict p_cursor,
			char *restrict file_end, char *restrict output)
{
	if (!p_cursor || !output) {
		perror("Invalid parameters for fasta extraction");
		return 0;
	}

	char *cursor = *p_cursor;
	if (!cursor || cursor >= file_end || *cursor != '>') {
		*p_cursor = cursor;
		return 0;
	}

	cursor = fasta_skip_line(cursor, file_end);
	char *write_pos = output;
	u64 length = 0;

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
