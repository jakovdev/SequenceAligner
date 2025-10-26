#include "core/io/format/csv.h"

#include <limits.h>
#include <stdint.h>

#include "system/arch.h"
#include "util/benchmark.h"
#include "util/print.h"

static inline size_t
csv_column_count(const char* line)
{
    if (!line || !*line || *line == '\n' || *line == '\r')
    {
        return 0;
    }

    size_t count = 1; // Start with 1 for the first column

    while (*line)
    {
        if (*line == ',')
        {
            count++;
        }

        else if (*line == '\n' || *line == '\r')
        {
            break;
        }

        line++;
    }

    return count;
}

static inline char*
csv_column_copy(const char* restrict file_start, const char* restrict file_end)
{
    ptrdiff_t delta = file_end - file_start;
    size_t len = delta < 0 ? 0 : (size_t)delta;
    char* name = MALLOC(name, len + 1);
    if (name)
    {
        memcpy(name, file_start, len);
        name[len] = '\0';
    }

    return name;
}

static inline void
csv_column_sequence(char** headers, size_t num_cols, size_t* seq_col)
{
    if (!headers || num_cols == 0 || num_cols > INT_MAX)
    {
        *seq_col = SIZE_MAX;
        return;
    }

    const char* seq_keywords[] = { "sequence", "seq",   "protein", "dna",
                                   "rna",      "amino", "peptide", "chain" };
    const size_t num_keywords = sizeof(seq_keywords) / sizeof(*seq_keywords);

    // First pass: exact match for sequence column
    for (size_t column = 0; column < num_cols; column++)
    {
        for (size_t key = 0; key < num_keywords; key++)
        {
            if (strcasecmp(headers[column], seq_keywords[key]) == 0)
            {
                *seq_col = column;
                return;
            }
        }
    }

    // Second pass: partial match (contains sequence keyword)
    for (size_t column = 0; column < num_cols; column++)
    {
        for (size_t key = 0; key < num_keywords; key++)
        {
            if (strcasestr(headers[column], seq_keywords[key]))
            {
                *seq_col = column;
                return;
            }
        }
    }

    // No match found
    *seq_col = SIZE_MAX;
}

char*
csv_header_parse(char* restrict file_cursor,
                 char* restrict file_end,
                 bool* no_header,
                 size_t* seq_col)
{
    print_error_prefix("CSV");

    char* header_start = file_cursor;
    *seq_col = SIZE_MAX;
    *no_header = false;

    size_t num_columns = 0;
    char** headers = NULL;

    num_columns = csv_column_count(header_start);
    print(VERBOSE, MSG_NONE, "Found %zu column(s) in input file", num_columns);

    if (!num_columns)
    {
        print(ERROR, MSG_NONE, "Invalid header (do you have an empty line or file?)");
        exit(1);
    }

    headers = MALLOC(headers, num_columns);

    if (!headers)
    {
        print(ERROR, MSG_NONE, "Memory allocation failed for column headers");
        exit(1);
    }

    for (size_t i = 0; i < num_columns; i++)
    {
        headers[i] = NULL;
    }

    const char* col_start = header_start;
    size_t column = 0;

    while (file_cursor < file_end)
    {
        if (*file_cursor == ',' || *file_cursor == '\n' || *file_cursor == '\r')
        {
            if (column < num_columns)
            {
                headers[column] = csv_column_copy(col_start, file_cursor);
                column++;
            }

            if (*file_cursor == ',' && column < num_columns)
            {
                col_start = file_cursor + 1;
            }

            else if (*file_cursor == '\n')
            {
                file_cursor++; // Move past newline
                break;
            }

            else if (*file_cursor == '\r')
            {
                file_cursor++; // Move past CR
                if (file_cursor < file_end && *file_cursor == '\n')
                {
                    file_cursor++; // Move past LF if present
                }

                break;
            }
        }

        file_cursor++;
    }

    csv_column_sequence(headers, num_columns, seq_col);

    // If auto-detection failed
    if (*seq_col == SIZE_MAX)
    {
        bench_io_end();
        char** choices = MALLOC(choices, num_columns + 2);

        if (!choices)
        {
            print(ERROR, MSG_NONE, "Memory allocation failed for choices array");
            if (headers)
            {
                for (size_t i = 0; i < num_columns; i++)
                {
                    if (headers[i])
                    {
                        free(headers[i]);
                    }
                }
                free(headers);
            }

            exit(1);
        }

        for (column = 0; column < num_columns; column++)
        {
            choices[column] = headers[column];
        }

        choices[num_columns] = "My csv file does not have a header! Do not skip it!";
        size_t choice_num = num_columns + 1;

        print(INFO, MSG_LOC(FIRST), "Could not automatically detect the sequence column.");
        print(INFO, MSG_LOC(MIDDLE), "Which column contains your sequences?");
        print(INFO, MSG_LOC(LAST), "Select the header name (this first line will be skipped!):");

        *seq_col = print(CHOICE, MSG_CHOICE(choices, choice_num), "Enter column number");
        free(choices);
        bench_io_start();
    }

    if (*seq_col == num_columns)
    {
        bench_io_end();
        print(INFO, MSG_LOC(LAST), "OK, select the column that displays a sequence");
        char** choices = headers;
        size_t choice_num = num_columns;
        *seq_col = print(CHOICE, MSG_CHOICE(choices, choice_num), "Enter column number");
        *no_header = true;
        bench_io_start();
    }

    if (*seq_col < num_columns && headers && headers[*seq_col])
    {
        const char* sequence_column = headers[*seq_col];
        print(VERBOSE, MSG_NONE, "Using column #%zu ('%s')", *seq_col + 1, sequence_column);
    }

    if (headers)
    {
        for (column = 0; column < num_columns; column++)
        {
            if (headers[column])
            {
                free(headers[column]);
            }
        }

        free(headers);
        headers = NULL;
    }

    return file_cursor;
}

bool
csv_line_next(char* restrict* restrict p_cursor)
{
    char* cursor = *p_cursor;

    while (*cursor && (*cursor == ' ' || *cursor == '\r' || *cursor == '\n'))
    {
        cursor++;
    }

    if (!*cursor)
    {
        *p_cursor = cursor;
        return false;
    }

    while (*cursor && *cursor != '\n' && *cursor != '\r')
    {
        cursor++;
    }

    while (*cursor && (*cursor == '\n' || *cursor == '\r'))
    {
        cursor++;
    }

    *p_cursor = cursor;
    return true;
}

size_t
csv_total_lines(char* restrict file_cursor, char* restrict file_end)
{
    size_t total_lines = 0;

    while (file_cursor < file_end && *file_cursor)
    {
        if (csv_line_next(&file_cursor))
        {
            total_lines++;
        }
    }

    return total_lines;
}

size_t
csv_line_column_extract(char* restrict* restrict p_cursor,
                        char* restrict output,
                        size_t target_column)
{
    char* cursor = *p_cursor;
    char* write_pos = NULL;
    size_t column = 0;
    size_t column_length = 0;

    while (*cursor && (*cursor == ' ' || *cursor == '\r' || *cursor == '\n'))
    {
        cursor++;
    }

    while (*cursor && *cursor != '\n' && *cursor != '\r')
    {
        if (column == target_column)
        {
            write_pos = output;

            while (*cursor && *cursor != ',' && *cursor != '\n' && *cursor != '\r')
            {
                *write_pos++ = *cursor++;
            }

            *write_pos = '\0';
            ptrdiff_t delta = write_pos - output;
            column_length = delta < 0 ? 0 : (size_t)delta;
        }

        else
        {
            // Skip other columns
            while (*cursor && *cursor != ',' && *cursor != '\n' && *cursor != '\r')
            {
                cursor++;
            }
        }

        if (*cursor == ',')
        {
            cursor++;
            column++;
        }
    }

    while (*cursor && (*cursor == '\n' || *cursor == '\r'))
    {
        cursor++;
    }

    *p_cursor = cursor;
    return column_length;
}

size_t
csv_line_column_length(char* cursor, size_t target_column)
{
    size_t column = 0;
    size_t column_length = 0;

    while (*cursor && (*cursor == ' ' || *cursor == '\r' || *cursor == '\n'))
    {
        cursor++;
    }

    while (*cursor && *cursor != '\n' && *cursor != '\r')
    {
        if (column == target_column)
        {
            char* col_start = cursor;
            while (*cursor && *cursor != ',' && *cursor != '\n' && *cursor != '\r')
            {
                cursor++;
            }

            ptrdiff_t delta = cursor - col_start;
            column_length = delta < 0 ? 0 : (size_t)delta;
            break;
        }

        else
        {
            while (*cursor && *cursor != ',' && *cursor != '\n' && *cursor != '\r')
            {
                cursor++;
            }
        }

        if (*cursor == ',')
        {
            cursor++;
            column++;
        }
    }

    return column_length;
}

bool
csv_validate(const char* restrict file_start, const char* restrict file_end)
{
    if (!file_start || !file_end || file_start >= file_end)
    {
        print(ERROR, MSG_NONE, "Invalid file bounds for validation");
        return false;
    }

    const char* cursor = file_start;
    size_t line_number = 0;
    size_t expected_columns = 0;
    bool first_line = true;

    while (cursor < file_end)
    {
        const char* line_start = cursor;
        while (cursor < file_end && (*cursor == ' ' || *cursor == '\t' || *cursor == '\r'))
        {
            cursor++;
        }

        if (cursor < file_end && *cursor == '\n')
        {
            cursor++;
            line_number++;
            continue;
        }

        if (cursor >= file_end)
        {
            break;
        }

        size_t current_columns = 0;
        bool field_started = false;

        while (cursor < file_end)
        {
            char ch = *cursor;

            if (ch == ',' || ch == '\n' || ch == '\r')
            {
                current_columns++;
                field_started = false;

                if (ch == '\n' || ch == '\r')
                {
                    if (ch == '\r' && (cursor + 1 < file_end) && cursor[1] == '\n')
                    {
                        cursor++;
                    }

                    cursor++;
                    line_number++;
                    break;
                }
            }

            else if (ch == '\0')
            {
                print(ERROR, MSG_NONE, "Null character found on line %zu", line_number);
                return false;
            }

            else
            {
                field_started = true;
            }

            cursor++;
        }

        if (field_started || (cursor > line_start && cursor > file_start && *(cursor - 1) == ','))
        {
            current_columns++;
        }

        if (first_line)
        {
            if (current_columns == 0)
            {
                print(ERROR, MSG_NONE, "Header line has zero columns");
                return false;
            }

            expected_columns = current_columns;
            first_line = false;
        }

        else if (current_columns > 0)
        {
            if (current_columns != expected_columns)
            {
                print(ERROR,
                      MSG_NONE,
                      "Expected %zu column(s), found %zu on line %zu",
                      expected_columns,
                      current_columns,
                      line_number);
                return false;
            }
        }
    }

    return true;
}
