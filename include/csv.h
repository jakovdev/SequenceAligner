#ifndef CSV_H
#define CSV_H

#include "files.h"
#include "print.h"
#include "benchmark.h"

typedef struct {
    const char* seq;
    size_t len;
} Data;

typedef struct {
    int seq_col_index;
    int num_columns;
    char** column_headers;
} CsvMetadata;

static CsvMetadata g_csv_metadata = {-1, 0, NULL};

INLINE char* skip_header(char* restrict current, char* restrict end) {
    while (current < end) {
        if (*current == '\n') {
            return current + 1;
        }
        current++;
    }
    return current;
}

INLINE void free_csv_metadata(void) {
    if (g_csv_metadata.column_headers) {
        for (int i = 0; i < g_csv_metadata.num_columns; i++) {
            if (g_csv_metadata.column_headers[i]) {
                free(g_csv_metadata.column_headers[i]);
            }
        }
        free(g_csv_metadata.column_headers);
        g_csv_metadata.column_headers = NULL;
    }
}

INLINE int count_columns(const char* line) {
    if (!line || !*line || *line == '\n' || *line == '\r') return 0;
    int count = 1;  // Start with 1 for the first column
    
    while (*line) {
        if (*line == ',') {
            count++;
        } else if (*line == '\n' || *line == '\r') {
            break;
        }
        line++;
    }
    
    return count;
}

INLINE char* copy_column_name(const char* start, const char* end) {
    size_t len = end - start;
    char* name = (char*)malloc(len + 1);
    if (name) {
        memcpy(name, start, len);
        name[len] = '\0';
    }
    return name;
}

INLINE int detect_sequence_column(char** headers, int num_cols) {
    const char* seq_keywords[] = {
        "sequence", "seq", "protein", "dna", "rna", "amino", "peptide", "chain"
    };
    const int num_keywords = sizeof(seq_keywords) / sizeof(seq_keywords[0]);
    
    // First pass: exact match for sequence column
    for (int i = 0; i < num_cols; i++) {
        for (int k = 0; k < num_keywords; k++) {
            if (strcasecmp(headers[i], seq_keywords[k]) == 0) {
                return i;
            }
        }
    }
    
    // Second pass: partial match (contains sequence keyword)
    for (int i = 0; i < num_cols; i++) {
        for (int k = 0; k < num_keywords; k++) {
            if (strcasestr(headers[i], seq_keywords[k])) {
                return i;
            }
        }
    }
    
    // No match found
    return -1;
}

INLINE void read_line_input(char* input_buffer, int buffer_size, int* choice, 
                            const char* prompt, size_t prompt_len) {
    int idx = 0;
    int c;
    
    apply_box_color();
    printf("%s", BOX_VERTICAL);
    reset_color();
    printf(" %s", prompt);
    
    fflush(stdout);
    
    // Read input character by character
    while (1) {
        c = getchar();
        
        // Exit on Enter key
        if (c == '\n' || c == '\r') {
            break;
        }
        
        // Only accept digits and limit input length
        if (isdigit(c) && idx < buffer_size - 1) {
            input_buffer[idx++] = c;
            input_buffer[idx] = '\0';
            printf("%c", c); // Echo the character
            fflush(stdout);
        }
    }
    
    *choice = atoi(input_buffer);
    int padding = OUTPUT_WIDTH + 8 - prompt_len - idx;
    for (int i = 0; i < padding; i++) printf(" ");

    apply_box_color();
    printf("%s", BOX_VERTICAL);
    reset_color();
}

INLINE void prompt_for_sequence_column(void) {
    // Pause the init timer before asking for user input
    double saved_time = bench_pause_init();
    
    print_info("Could not automatically detect the sequence column.");
    print_info("Please select the column containing sequence data:");
    
    for (int i = 0; i < g_csv_metadata.num_columns; i++) {
        size_t option_text_len = strlen(g_csv_metadata.column_headers[i]) + 3;
        int padding = OUTPUT_WIDTH - 3 - option_text_len;
        if (padding < 0) padding = 0;

        apply_box_color();
        printf("%s", BOX_VERTICAL);
        reset_color();

        printf(" %s%d:%s %s", ANSI_COLOR_CYAN, i + 1, ANSI_COLOR_RESET, g_csv_metadata.column_headers[i]);

        for (int j = 0; j < padding; j++) printf(" ");

        apply_box_color();
        printf("%s\n", BOX_VERTICAL);
        reset_color();
    }
    
    char prompt[128];
    snprintf(prompt, sizeof(prompt), "%s%s %sEnter column number (1-%d): ", ANSI_COLOR_BLUE, ICON_INFO, ANSI_COLOR_RESET, g_csv_metadata.num_columns);
    size_t prompt_len = strlen(prompt);
    
    char input_buffer[32] = {0};
    int choice = -1;
    #ifdef _WIN32
    HANDLE hStdin = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode;
    GetConsoleMode(hStdin, &mode);
    SetConsoleMode(hStdin, mode & ~(ENABLE_ECHO_INPUT | ENABLE_LINE_INPUT));
    #else
    struct termios old_term, new_term;
    tcgetattr(STDIN_FILENO, &old_term);
    new_term = old_term;
    new_term.c_lflag &= ~(ICANON | ECHO); // Disable canonical mode and echo
    tcsetattr(STDIN_FILENO, TCSANOW, &new_term);
    #endif

    // Display input prompt and get user input in the same line
    read_line_input(input_buffer, sizeof(input_buffer), &choice, prompt, prompt_len);
    printf("\n"); // Manually add the newline for consistency

    #ifdef _WIN32
    SetConsoleMode(hStdin, mode);
    #else
    tcsetattr(STDIN_FILENO, TCSANOW, &old_term);
    #endif

    // Validate input
    while (choice < 1 || choice > g_csv_metadata.num_columns) {
        print_warning("Invalid input! Please enter a number between 1 and %d.", g_csv_metadata.num_columns);
        memset(input_buffer, 0, sizeof(input_buffer));

        #ifdef _WIN32
        SetConsoleMode(hStdin, mode & ~(ENABLE_ECHO_INPUT | ENABLE_LINE_INPUT));
        #else
        tcsetattr(STDIN_FILENO, TCSANOW, &new_term);
        #endif

        read_line_input(input_buffer, sizeof(input_buffer), &choice, prompt, prompt_len);
        printf("\n");

        #ifdef _WIN32
        SetConsoleMode(hStdin, mode);
        #else
        tcsetattr(STDIN_FILENO, TCSANOW, &old_term);
        #endif
    }
    
    g_csv_metadata.seq_col_index = choice - 1;
    print_success("Selected column %d: %s", choice, g_csv_metadata.column_headers[choice - 1]);
    bench_resume_init(saved_time);
}

INLINE char* parse_header(char* restrict current, char* restrict end) {
    char* header_start = current;
    
    g_csv_metadata.num_columns = count_columns(header_start);
    
    if (g_csv_metadata.num_columns <= 0) {
        print_error("Invalid CSV header");
        print_step_header_end(1);
        exit(1);
    }
    
    g_csv_metadata.column_headers = (char**)malloc(g_csv_metadata.num_columns * sizeof(char*));
    if (!g_csv_metadata.column_headers) {
        print_error("Memory allocation failed for column headers");
        print_step_header_end(1);
        exit(1);
    }
    
    const char* col_start = header_start;
    int col_idx = 0;
    
    while (current < end) {
        if (*current == ',' || *current == '\n' || *current == '\r') {
            if (col_idx < g_csv_metadata.num_columns) {
                g_csv_metadata.column_headers[col_idx] = copy_column_name(col_start, current);
                col_idx++;
            }
            
            if (*current == ',' && col_idx < g_csv_metadata.num_columns) {
                col_start = current + 1;
            } else if (*current == '\n') {
                current++; // Move past newline
                break;
            } else if (*current == '\r') {
                current++; // Move past CR
                if (current < end && *current == '\n') {
                    current++; // Move past LF if present
                }
                break;
            }
        }
        current++;
    }
    
    g_csv_metadata.seq_col_index = detect_sequence_column(g_csv_metadata.column_headers, g_csv_metadata.num_columns);
    
    // If auto-detection failed
    if (g_csv_metadata.seq_col_index < 0) {
        prompt_for_sequence_column();
    }
    
    print_verbose("Detected %d columns in CSV", g_csv_metadata.num_columns);
    print_verbose("Using column %d ('%s') for sequences", 
                  g_csv_metadata.seq_col_index + 1, 
                  g_csv_metadata.column_headers[g_csv_metadata.seq_col_index]);
    
    return current;
}

INLINE size_t count_csv_line(char** current) {
    char* p = *current;
    
    while (*p && (*p == ' ' || *p == '\r' || *p == '\n')) p++;
    
    if (!*p) {
        *current = p;
        return 0;
    }

    #ifdef USE_SIMD
    const veci_t nl_vec = set1_epi8('\n');
    const veci_t cr_vec = set1_epi8('\r');
    
    while (*p) {
        veci_t data = loadu((veci_t*)p);
        
        #if defined(__AVX512F__) && defined(__AVX512BW__)
        num_t mask_nl = cmpeq_epi8(data, nl_vec);
        num_t mask_cr = cmpeq_epi8(data, cr_vec);
        num_t mask = or_mask(mask_nl, mask_cr);
        #else
        veci_t is_newline = or_si(
            cmpeq_epi8(data, nl_vec),
            cmpeq_epi8(data, cr_vec)
        );
        num_t mask = movemask_epi8(is_newline);
        #endif
        
        if (mask) {
            num_t pos = ctz(mask);
            p += pos;
            break;
        }
        
        p += BYTES;
    }
    #else
    while (*p && *p != '\n' && *p != '\r') {
        p++;
    }
    #endif
    
    while (*p && (*p == '\n' || *p == '\r')) p++;
    *current = p;
    return 1;
}

INLINE size_t parse_csv_line(char** current, char* seq) {
    char* p = *current;
    char* write_pos = NULL;
    size_t col = 0;
    size_t seq_len = 0;

    while (*p && (*p == ' ' || *p == '\r' || *p == '\n')) p++;

    #ifdef USE_SIMD
    const veci_t delim_vec = set1_epi8(',');
    const veci_t nl_vec = set1_epi8('\n');
    const veci_t cr_vec = set1_epi8('\r');

    while (*p && *p != '\n' && *p != '\r') {
        if ((int)col == g_csv_metadata.seq_col_index) {
            write_pos = seq;
            while (*p && *p != ',' && *p != '\n' && *p != '\r') {
                veci_t data = loadu((veci_t*)p);
                
                #if defined(__AVX512F__) && defined(__AVX512BW__)
                num_t mask_delim = cmpeq_epi8(data, delim_vec);
                num_t mask_nl = cmpeq_epi8(data, nl_vec);
                num_t mask_cr = cmpeq_epi8(data, cr_vec);
                num_t mask = or_mask(or_mask(mask_delim, mask_nl), mask_cr);
                #else
                veci_t is_delim = or_si(
                    or_si(
                        cmpeq_epi8(data, delim_vec),
                        cmpeq_epi8(data, nl_vec)
                    ),
                    cmpeq_epi8(data, cr_vec)
                );
                num_t mask = movemask_epi8(is_delim);
                #endif

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
        if ((int)col == g_csv_metadata.seq_col_index) {
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