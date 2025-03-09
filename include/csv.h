#ifndef CSV_H
#define CSV_H

#include "files.h"

typedef struct {
    const char* seq;
    const char* other_data;
    size_t len;
} Data;

#if MODE_WRITE == 1
typedef struct {
    const char* parts[3];
    size_t lengths[3]; 
    size_t data_pos[2 * READ_CSV_COLS - 2 + SIMILARITY_ANALYSIS * 4];
    int8_t col_to_data[2 * READ_CSV_COLS + 2 + SIMILARITY_ANALYSIS * 4];
} Format;

static Format fmt = {
    .parts = {NULL},
    .lengths = {0},
    .data_pos = {0},
    .col_to_data = {-1}
};
#endif

INLINE char* skip_header(char* current, char* end) {
    while (current < end) {
        if (*current == '\n') {
            return current + 1;
        }
        current++;
    }
    return current;
}

INLINE void init_format(void) {
    #if MODE_WRITE == 1
    const char* format = WRITE_CSV_ALIGN_FMT;
    size_t part = 0;
    fmt.parts[part] = format;
    
    while (*format) {
        if (format[0] == '%' && format[1] == 's') {
            fmt.lengths[part] = format - fmt.parts[part];
            format += 2;
            part++;
            fmt.parts[part] = format;
        } else {
            format++;
        }
    }
    fmt.lengths[part] = format - fmt.parts[part];

    memset(fmt.col_to_data, -1, sizeof(fmt.col_to_data));

    bool write_used[2 * READ_CSV_COLS + 2 + SIMILARITY_ANALYSIS * 4] = {false};
    write_used[WRITE_CSV_SEQ1_POS] = true;
    write_used[WRITE_CSV_SEQ1_POS + 1] = true;
    write_used[WRITE_CSV_SCORE_POS] = true;
    write_used[WRITE_CSV_ALIGN_POS] = true;
    #if SIMILARITY_ANALYSIS == 1
    write_used[WRITE_CSV_MATCHES_POS] = true;
    write_used[WRITE_CSV_MISMATCHES_POS] = true;
    write_used[WRITE_CSV_GAPS_POS] = true;
    write_used[WRITE_CSV_SIMILARITY_POS] = true;
    #endif
    
    size_t idx = 0;
    for(size_t i = 0; i < 2 * READ_CSV_COLS + 2 + SIMILARITY_ANALYSIS * 4 && idx < (READ_CSV_COLS - 1); i++) {
        if(!write_used[i]) {
            fmt.data_pos[idx] = i;
            fmt.col_to_data[i] = idx;
            fmt.col_to_data[i + 1] = idx;
            write_used[i] = write_used[i + 1] = true;
            idx++;
            i++;
        }
    }
    #endif
}

#if MODE_WRITE == 1
INLINE size_t buffer_output(char buffer[WRITE_BUF], size_t pos, const Data* restrict prev, const Data* restrict curr, const Alignment* restrict result) {
    char* buf = &buffer[pos];
    char* start = buf;
    
    for (size_t col = 0; col < 4 + (READ_CSV_COLS - 1) * 2 + SIMILARITY_ANALYSIS * 4; col++) {
        if (col > 0) *buf++ = ',';
        int8_t data_idx = fmt.col_to_data[col];
        if (data_idx >= 0) {
            const char* str = (col == fmt.data_pos[data_idx]) ? prev->other_data : curr->other_data;
            while(*str) *buf++ = *str++;
            continue;
        }
        switch (col) {
            case WRITE_CSV_SEQ1_POS:
                buf = fast_strcpy(buf, prev->seq, prev->len);
                break;
            case (WRITE_CSV_SEQ1_POS + 1):
                buf = fast_strcpy(buf, curr->seq, curr->len);
                break;
            case WRITE_CSV_SCORE_POS:
                buf = int_to_str(buf, result->score);
                break;
            case WRITE_CSV_ALIGN_POS:
                buf = fast_strcpy(buf, fmt.parts[0], fmt.lengths[0]);
                buf = fast_strcpy(buf, result->seq1_aligned, strlen(result->seq1_aligned));
                buf = fast_strcpy(buf, fmt.parts[1], fmt.lengths[1]);
                buf = fast_strcpy(buf, result->seq2_aligned, strlen(result->seq2_aligned));
                buf = fast_strcpy(buf, fmt.parts[2], fmt.lengths[2]);
                break;
            #if SIMILARITY_ANALYSIS == 1
            case WRITE_CSV_MATCHES_POS:
                buf = int_to_str(buf, result->matches);
                break;
            case WRITE_CSV_MISMATCHES_POS:
                buf = int_to_str(buf, result->mismatches);
                break;
            case WRITE_CSV_GAPS_POS:
                buf = int_to_str(buf, result->gaps);
                break;
            case WRITE_CSV_SIMILARITY_POS: {
                int percentage = (int)(result->similarity * 10000);
                buf = int_to_str(buf, percentage / 100);
                *buf++ = '.';
                int decimal = percentage % 100;
                if (decimal < 10) *buf++ = '0';
                buf = int_to_str(buf, decimal);
                *buf++ = '%';
                break;
            }
            #endif
        }
    }
    
    *buf++ = '\n';
    return buf - start;
}
#endif

INLINE size_t parse_csv_line(char** restrict current, 
                            char seq[MAX_SEQ_LEN],
                            char other_data[MAX_CSV_LINE - MAX_SEQ_LEN]) {
    char* p = *current;
    char* write_pos = NULL;
    size_t col = 0;
    size_t data_write_pos = 0;
    size_t seq_len = 0;

    while (*p && (*p == ' ' || *p == '\r' || *p == '\n')) p++;

    #ifdef USE_AVX
    const veci_t delim_vec = set1_epi8(',');
    const veci_t nl_vec = set1_epi8('\n');
    const veci_t cr_vec = set1_epi8('\r');

    while (*p && *p != '\n' && *p != '\r') {
        write_pos = (col == READ_CSV_SEQ_POS) ? seq : other_data + data_write_pos;
        while (*p && *p != ',' && *p != '\n' && *p != '\r') {
            veci_t data = loadu((veci_t*)p);
            veci_t is_delim = or_si(
                or_si(
                    cmpeq_epi8(data, delim_vec),
                    cmpeq_epi8(data, nl_vec)
                ),
                cmpeq_epi8(data, cr_vec)
            );
            num_t mask = movemask_epi8(is_delim);

            if (mask) {
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
            
            PREFETCH(p + MAX_CSV_LINE);
        }
    #else
    while (*p && *p != '\n' && *p != '\r') {
        write_pos = (col == READ_CSV_SEQ_POS) ? seq : other_data + data_write_pos;
        
        while (*p && *p != ',' && *p != '\n' && *p != '\r') {
            *write_pos++ = *p++;
        }
    #endif
        *write_pos = '\0';
        
        if (col == READ_CSV_SEQ_POS) {
            seq_len = write_pos - seq;
        } else {
            data_write_pos = write_pos - other_data + 1;
        }
        
        if (*p == ',') { p++; col++; }
    }
    
    while (*p && (*p == '\n' || *p == '\r')) p++;
    *current = p;
    return seq_len;
}

#endif