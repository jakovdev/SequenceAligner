#pragma once
#ifndef CSV_H
#define CSV_H

#include "arch.h"
#include "print.h"

static inline int
csv_column_count(const char* line)
{
    if (!line || !*line || *line == '\n' || *line == '\r')
    {
        return 0;
    }

    int count = 1; // Start with 1 for the first column

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
    size_t len = file_end - file_start;
    char* name = MALLOC(name, len + 1);
    if (name)
    {
        memcpy(name, file_start, len);
        name[len] = '\0';
    }

    return name;
}

static inline int
csv_column_sequence(char** headers, int num_cols)
{
    const char* seq_keywords[] = { "sequence", "seq",   "protein", "dna",
                                   "rna",      "amino", "peptide", "chain" };

    const int num_keywords = sizeof(seq_keywords) / sizeof(seq_keywords[0]);

    // First pass: exact match for sequence column
    for (int column = 0; column < num_cols; column++)
    {
        for (int key = 0; key < num_keywords; key++)
        {
            if (strcasecmp(headers[column], seq_keywords[key]) == 0)
            {
                return column;
            }
        }
    }

    // Second pass: partial match (contains sequence keyword)
    for (int column = 0; column < num_cols; column++)
    {
        for (int key = 0; key < num_keywords; key++)
        {
            if (strcasestr(headers[column], seq_keywords[key]))
            {
                return column;
            }
        }
    }

    // No match found
    return -1;
}

static inline char*
csv_header_parse(char* restrict file_cursor, char* restrict file_end, bool* no_header, int* seq_col)
{
    char* header_start = file_cursor;
    *seq_col = -1;
    *no_header = false;

    int num_columns = 0;
    char** headers = NULL;

    num_columns = csv_column_count(header_start);
    print(VERBOSE, MSG_NONE, "Found %d columns in input file", num_columns);

    if (num_columns <= 0)
    {
        print(ERROR, MSG_NONE, "CSV | Invalid header (do you have an empty line or file?)");
        exit(1);
    }

    headers = MALLOC(headers, num_columns);

    if (!headers)
    {
        print(ERROR, MSG_NONE, "CSV | Memory allocation failed for column headers");
        exit(1);
    }

    const char* col_start = header_start;
    int column = 0;

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

    *seq_col = csv_column_sequence(headers, num_columns);

    // If auto-detection failed
    if (*seq_col < 0)
    {
        char** choices = MALLOC(choices, num_columns + 2);
        for (column = 0; column < num_columns; column++)
        {
            choices[column] = headers[column];
        }

        choices[num_columns] = "My csv file does not have a header! Do not skip it!";
        int choice_num = num_columns + 1;

        print(INFO, MSG_LOC(FIRST), "Could not automatically detect the sequence column.");
        print(INFO, MSG_LOC(MIDDLE), "Which column contains your sequences?");
        print(INFO, MSG_LOC(LAST), "Select the header name (this first line will be skipped!):");

        *seq_col = print(CHOICE, MSG_CHOICE(choices, choice_num), "Enter column number");
    }

    if (*seq_col == num_columns)
    {
        print(INFO, MSG_LOC(LAST), "OK, select the column that displays a sequence");
        char** choices = headers;
        size_t choice_num = num_columns;
        *seq_col = print(CHOICE, MSG_CHOICE(choices, choice_num), "Enter column number");
        *no_header = true;
    }

    const char* sequence_column = headers[*seq_col];
    print(VERBOSE, MSG_NONE, "Using column %d ('%s') for sequences", *seq_col + 1, sequence_column);

    if (headers)
    {
        for (int column = 0; column < num_columns; column++)
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

static inline bool
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

#ifdef USE_SIMD
    const veci_t nl_vec = set1_epi8('\n');
    const veci_t cr_vec = set1_epi8('\r');

    while (*cursor)
    {
        veci_t data = loadu((veci_t*)cursor);

#if defined(__AVX512F__) && defined(__AVX512BW__)
        num_t mask_nl = cmpeq_epi8(data, nl_vec);
        num_t mask_cr = cmpeq_epi8(data, cr_vec);
        num_t mask = or_mask(mask_nl, mask_cr);
#else
        veci_t is_newline = or_si(cmpeq_epi8(data, nl_vec), cmpeq_epi8(data, cr_vec));
        num_t mask = movemask_epi8(is_newline);
#endif

        if (mask)
        {
            num_t pos = ctz(mask);
            cursor += pos;
            break;
        }

        cursor += BYTES;
    }

#else
    while (*cursor && *cursor != '\n' && *cursor != '\r')
    {
        cursor++;
    }

#endif

    while (*cursor && (*cursor == '\n' || *cursor == '\r'))
    {
        cursor++;
    }

    *p_cursor = cursor;
    return true;
}

static inline size_t
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

static inline size_t
csv_line_column_extract(char* restrict* restrict p_cursor, char* restrict output, int target_column)
{
    char* cursor = *p_cursor;
    char* write_pos = NULL;
    int column = 0;
    size_t column_length = 0;

    while (*cursor && (*cursor == ' ' || *cursor == '\r' || *cursor == '\n'))
    {
        cursor++;
    }

#ifdef USE_SIMD
    const veci_t delim_vec = set1_epi8(',');
    const veci_t nl_vec = set1_epi8('\n');
    const veci_t cr_vec = set1_epi8('\r');

    while (*cursor && *cursor != '\n' && *cursor != '\r')
    {
        if (column == target_column)
        {
            write_pos = output;
            while (*cursor && *cursor != ',' && *cursor != '\n' && *cursor != '\r')
            {
                veci_t data = loadu((veci_t*)cursor);

#if defined(__AVX512F__) && defined(__AVX512BW__)
                num_t mask_delim = cmpeq_epi8(data, delim_vec);
                num_t mask_nl = cmpeq_epi8(data, nl_vec);
                num_t mask_cr = cmpeq_epi8(data, cr_vec);
                num_t mask = or_mask(or_mask(mask_delim, mask_nl), mask_cr);
#else
                veci_t is_delim = or_si(
                    or_si(cmpeq_epi8(data, delim_vec), cmpeq_epi8(data, nl_vec)),
                    cmpeq_epi8(data, cr_vec));
                num_t mask = movemask_epi8(is_delim);
#endif

                if (mask)
                {
                    num_t pos = ctz(mask);
                    storeu((veci_t*)write_pos, data);
                    write_pos[pos] = '\0';
                    write_pos += pos;
                    cursor += pos;
                    break;
                }

                storeu((veci_t*)write_pos, data);
                cursor += BYTES;
                write_pos += BYTES;
            }

            *write_pos = '\0';
            column_length = write_pos - output;
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

#else
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
            column_length = write_pos - output;
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

#endif

    while (*cursor && (*cursor == '\n' || *cursor == '\r'))
    {
        cursor++;
    }

    *p_cursor = cursor;
    return column_length;
}

#endif