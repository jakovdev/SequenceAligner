#ifndef CSV_H
#define CSV_H

#include "print.h"

typedef struct
{
    int num_columns;
    char** column_headers;
} CsvMetadata;

static int g_seq_col_index = -1;

INLINE char*
csv_header_skip(char* restrict current, char* restrict end)
{
    while (current < end)
    {
        if (*current == '\n')
        {
            return current + 1;
        }

        current++;
    }

    return current;
}

INLINE void
csv_metadata_free(CsvMetadata* csv_metadata)
{
    if (csv_metadata->column_headers)
    {
        for (int i = 0; i < csv_metadata->num_columns; i++)
        {
            if (csv_metadata->column_headers[i])
            {
                free(csv_metadata->column_headers[i]);
            }
        }

        free(csv_metadata->column_headers);
        csv_metadata->column_headers = NULL;
    }
}

INLINE int
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

INLINE char*
csv_column_copy(const char* start, const char* end)
{
    size_t len = end - start;
    char* name = malloc(len + 1);
    if (name)
    {
        memcpy(name, start, len);
        name[len] = '\0';
    }

    return name;
}

INLINE int
csv_column_sequence(char** headers, int num_cols)
{
    const char* seq_keywords[] = { "sequence", "seq",   "protein", "dna",
                                   "rna",      "amino", "peptide", "chain" };

    const int num_keywords = sizeof(seq_keywords) / sizeof(seq_keywords[0]);

    // First pass: exact match for sequence column
    for (int i = 0; i < num_cols; i++)
    {
        for (int k = 0; k < num_keywords; k++)
        {
            if (strcasecmp(headers[i], seq_keywords[k]) == 0)
            {
                return i;
            }
        }
    }

    // Second pass: partial match (contains sequence keyword)
    for (int i = 0; i < num_cols; i++)
    {
        for (int k = 0; k < num_keywords; k++)
        {
            if (strcasestr(headers[i], seq_keywords[k]))
            {
                return i;
            }
        }
    }

    // No match found
    return -1;
}

INLINE char*
csv_header_parse(char* restrict current, char* restrict end)
{
    char* header_start = current;

    __attribute__((cleanup(csv_metadata_free))) CsvMetadata csv_metadata = { 0, NULL };

    csv_metadata.num_columns = csv_column_count(header_start);

    if (csv_metadata.num_columns <= 0)
    {
        print(ERROR, MSG_NONE, "Invalid CSV header");
        print(SECTION, MSG_NONE, NULL);
        exit(1);
    }

    csv_metadata.column_headers = malloc(csv_metadata.num_columns *
                                         sizeof(*csv_metadata.column_headers));

    if (!csv_metadata.column_headers)
    {
        print(ERROR, MSG_NONE, "Memory allocation failed for column headers");
        print(SECTION, MSG_NONE, NULL);
        exit(1);
    }

    const char* col_start = header_start;
    int col_idx = 0;

    while (current < end)
    {
        if (*current == ',' || *current == '\n' || *current == '\r')
        {
            if (col_idx < csv_metadata.num_columns)
            {
                csv_metadata.column_headers[col_idx] = csv_column_copy(col_start, current);
                col_idx++;
            }

            if (*current == ',' && col_idx < csv_metadata.num_columns)
            {
                col_start = current + 1;
            }

            else if (*current == '\n')
            {
                current++; // Move past newline
                break;
            }

            else if (*current == '\r')
            {
                current++; // Move past CR
                if (current < end && *current == '\n')
                {
                    current++; // Move past LF if present
                }

                break;
            }
        }

        current++;
    }

    g_seq_col_index = csv_column_sequence(csv_metadata.column_headers, csv_metadata.num_columns);

    // If auto-detection failed
    if (g_seq_col_index < 0)
    {
        print(INFO, MSG_LOC(FIRST), "Could not automatically detect the sequence column.");
        print(INFO, MSG_LOC(LAST), "Please select the column containing sequence data:");
        g_seq_col_index = print(CHOICE,
                                MSG_PROMPT(csv_metadata.column_headers),
                                "Enter column number");
    }

    print(VERBOSE, MSG_LOC(FIRST), "Detected %d columns in CSV", csv_metadata.num_columns);
    print(VERBOSE,
          MSG_LOC(MIDDLE),
          "Using column %d ('%s') for sequences",
          g_seq_col_index + 1,
          csv_metadata.column_headers[g_seq_col_index]);

    return current;
}

INLINE size_t
csv_line_count(char** current)
{
    char* p = *current;

    while (*p && (*p == ' ' || *p == '\r' || *p == '\n'))
    {
        p++;
    }

    if (!*p)
    {
        *current = p;
        return 0;
    }

#ifdef USE_SIMD
    const veci_t nl_vec = set1_epi8('\n');
    const veci_t cr_vec = set1_epi8('\r');

    while (*p)
    {
        veci_t data = loadu((veci_t*)p);

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
            p += pos;
            break;
        }

        p += BYTES;
    }

#else
    while (*p && *p != '\n' && *p != '\r')
    {
        p++;
    }

#endif

    while (*p && (*p == '\n' || *p == '\r'))
    {
        p++;
    }

    *current = p;
    return 1;
}

INLINE size_t
csv_sequence_lines(char* current, char* end)
{
    size_t total_seqs = 0;

    while (current < end && *current)
    {
        if (csv_line_count(&current))
        {
            total_seqs++;
        }
    }

    return total_seqs;
}

INLINE size_t
csv_line_parse(char** current, char* seq)
{
    char* p = *current;
    char* write_pos = NULL;
    size_t col = 0;
    size_t seq_len = 0;

    while (*p && (*p == ' ' || *p == '\r' || *p == '\n'))
    {
        p++;
    }

#ifdef USE_SIMD
    const veci_t delim_vec = set1_epi8(',');
    const veci_t nl_vec = set1_epi8('\n');
    const veci_t cr_vec = set1_epi8('\r');

    while (*p && *p != '\n' && *p != '\r')
    {
        if ((int)col == g_seq_col_index)
        {
            write_pos = seq;
            while (*p && *p != ',' && *p != '\n' && *p != '\r')
            {
                veci_t data = loadu((veci_t*)p);

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
                    p += pos;
                    break;
                }

                storeu((veci_t*)write_pos, data);
                p += BYTES;
                write_pos += BYTES;
            }

            *write_pos = '\0';
            seq_len = write_pos - seq;
        }

        else
        {
            // Skip other columns
            while (*p && *p != ',' && *p != '\n' && *p != '\r')
            {
                p++;
            }
        }

        if (*p == ',')
        {
            p++;
            col++;
        }
    }

#else
    while (*p && *p != '\n' && *p != '\r')
    {
        if ((int)col == g_seq_col_index)
        {
            write_pos = seq;

            while (*p && *p != ',' && *p != '\n' && *p != '\r')
            {
                *write_pos++ = *p++;
            }

            *write_pos = '\0';
            seq_len = write_pos - seq;
        }

        else
        {
            // Skip other columns
            while (*p && *p != ',' && *p != '\n' && *p != '\r')
            {
                p++;
            }
        }

        if (*p == ',')
        {
            p++;
            col++;
        }
    }

#endif

    while (*p && (*p == '\n' || *p == '\r'))
    {
        p++;
    }

    *current = p;
    return seq_len;
}

#endif