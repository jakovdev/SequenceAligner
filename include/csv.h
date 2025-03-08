#ifndef CSV_H
#define CSV_H

#include "files.h"

typedef struct {
    const char* seq;
    size_t len;
} Data;

INLINE char* skip_header(char* restrict current, char* restrict end) {
    while (current < end) {
        if (*current == '\n') {
            return current + 1;
        }
        current++;
    }
    return current;
}

INLINE size_t parse_csv_line(char** current, 
                             char seq[MAX_SEQ_LEN]) {
    char* p = *current;
    char* write_pos = NULL;
    size_t col = 0;
    size_t seq_len = 0;

    while (*p && (*p == ' ' || *p == '\r' || *p == '\n')) p++;

    #ifdef USE_AVX
    const veci_t delim_vec = set1_epi8(',');
    const veci_t nl_vec = set1_epi8('\n');
    const veci_t cr_vec = set1_epi8('\r');

    while (*p && *p != '\n' && *p != '\r') {
        if (col == READ_CSV_SEQ_POS) {
            write_pos = seq;
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
            
            *write_pos = '\0';
            seq_len = write_pos - seq;
        } else {
            // Skip other columns
            while (*p && *p != ',' && *p != '\n' && *p != '\r') {
                p++;
            }
        }
        
        if (*p == ',') { p++; col++; }
    }
    #else
    while (*p && *p != '\n' && *p != '\r') {
        if (col == READ_CSV_SEQ_POS) {
            write_pos = seq;
            
            while (*p && *p != ',' && *p != '\n' && *p != '\r') {
                *write_pos++ = *p++;
            }
            
            *write_pos = '\0';
            seq_len = write_pos - seq;
        } else {
            // Skip other columns
            while (*p && *p != ',' && *p != '\n' && *p != '\r') {
                p++;
            }
        }
        
        if (*p == ',') { p++; col++; }
    }
    #endif
    
    while (*p && (*p == '\n' || *p == '\r')) p++;
    *current = p;
    return seq_len;
}

#endif