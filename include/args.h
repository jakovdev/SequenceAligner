#ifndef ARGS_H
#define ARGS_H

#include "common.h"
#include "print.h"

// Alignment methods (EXPANDABLE)
#define ALIGN_NEEDLEMAN_WUNSCH 0
#define ALIGN_GOTOH_AFFINE     1
#define ALIGN_SMITH_WATERMAN   2

// Scoring matrices (EXPANDABLE)
#define SCORE_BLOSUM50 0
#define SCORE_BLOSUM62 1

typedef struct {
    char input_file_path[MAX_PATH];
    char output_file_path[MAX_PATH];
    int align_method;           // 0=NW, 1=GA, 2=SW (EXPANDABLE)
    int scoring_matrix;         // 0=BLOSUM50, 1=BLOSUM62 (EXPANDABLE)
    int gap_penalty;            // Linear gap penalty (NW)
    int gap_start;              // Affine gap start penalty (GA|SW)
    int gap_extend;             // Affine gap extend penalty (GA|SW)
    int num_threads;            // Number of threads (0 = auto)
    int compression_level;      // HDF5 compression level (0-9)
    
    // Flags
    unsigned mode_benchmark : 1; // Enable benchmarking
    unsigned mode_write : 1;     // Enable output file writing
    unsigned mode_trim : 1;      // Enable sequence trimming
    #if MODE_CREATE_ALIGNED_STRINGS == 1
    unsigned aligned_strings : 1;// Create strings with gaps vs score-only
    #endif
    unsigned verbose : 1;        // Verbose output
    unsigned quiet : 1;          // Quiet output
} Args;

// Long options equivalents for getopt
static struct option long_options[] = {
    {"input",           required_argument, 0, 'i'},
    {"output",          required_argument, 0, 'o'},
    {"align",           required_argument, 0, 'a'},
    {"matrix",          required_argument, 0, 'm'},
    {"gap-penalty",     required_argument, 0, 'p'},
    {"gap-start",       required_argument, 0, 's'},
    {"gap-extend",      required_argument, 0, 'e'},
    {"threads",         required_argument, 0, 't'},
    {"compression",     required_argument, 0, 'z'},
    {"benchmark",       no_argument,       0, 'B'},
    {"no-write",        no_argument,       0, 'W'},
    {"trim",            no_argument,       0, 'T'},
    #if MODE_CREATE_ALIGNED_STRINGS == 1
    {"aligned-strings", no_argument,       0, 'A'},
    #endif
    {"verbose",         no_argument,       0, 'v'},
    {"quiet",           no_argument,       0, 'q'},
    {"help",            no_argument,       0, 'h'},
    {0, 0, 0, 0}
};

static Args g_args = {0};

INLINE void print_usage(const char* program_name) {
    printf("Usage: %s [OPTIONS]\n\n", program_name);
    printf("Sequence Alignment Tool - Fast pairwise sequence alignment\n\n");
    printf("Options:\n");
    printf("  -i, --input FILE       Input CSV file path [default: ./datasets/avpdb.csv]\n");
    printf("  -o, --output FILE      Output HDF5 file path [default: ./results/matrix.h5]\n");
    printf("  -a, --align METHOD     Alignment method [default: nw]\n");
    printf("                           nw: Needleman-Wunsch (global alignment)\n");
    printf("                           ga: Gotoh algorithm with affine gap penalty\n");
    printf("                           sw: Smith-Waterman (local alignment)\n");
    printf("  -m, --matrix MATRIX    Scoring matrix [default: blosum50]\n");
    printf("                           blosum50, blosum62\n");
    printf("  -p, --gap-penalty N    Linear gap penalty for NW [default: 4]\n");
    printf("  -s, --gap-start N      Affine gap start penalty for GA/SW [default: 10]\n");
    printf("  -e, --gap-extend N     Affine gap extend penalty for GA/SW [default: 1]\n");
    printf("  -t, --threads N        Number of threads (0 = auto) [default: 0]\n");
    printf("  -z, --compression N    HDF5 compression level (0-9) [default: 1]\n");
    printf("  -B, --benchmark        Enable benchmarking mode\n");
    printf("  -W, --no-write         Disable writing to output file\n");
    printf("  -T, --trim             Enable sequence trimming\n");
    #if MODE_CREATE_ALIGNED_STRINGS == 1
    printf("  -A, --aligned-strings  Create aligned strings with gaps (slower)\n");
    #endif
    printf("  -v, --verbose          Enable verbose output\n");
    printf("  -q, --quiet            Suppress all output\n");
    printf("  -h, --help             Display this help message\n");
}

INLINE int parse_alignment_method(const char* arg) {
    if (!arg) return ALIGN_NEEDLEMAN_WUNSCH;
    
    // (EXPANDABLE)
    if (isdigit(arg[0])) {
        int method = atoi(arg);
        return (method >= 0 && method <= 2) ? method : ALIGN_NEEDLEMAN_WUNSCH;
    }
    if (strcasecmp(arg, "nw") == 0 || strcasecmp(arg, "needleman") == 0) 
        return ALIGN_NEEDLEMAN_WUNSCH;
    if (strcasecmp(arg, "ga") == 0 || strcasecmp(arg, "gotoh") == 0 || 
        strcasecmp(arg, "affine") == 0)
        return ALIGN_GOTOH_AFFINE;
    if (strcasecmp(arg, "sw") == 0 || strcasecmp(arg, "smith") == 0)
        return ALIGN_SMITH_WATERMAN;
        
    return ALIGN_NEEDLEMAN_WUNSCH;
}

INLINE int parse_scoring_matrix(const char* arg) {
    if (!arg) return SCORE_BLOSUM50;

    // (EXPANDABLE)
    if (isdigit(arg[0])) {
        int matrix = atoi(arg);
        return (matrix >= 0 && matrix <= 1) ? matrix : SCORE_BLOSUM50;
    }
    if (strcasecmp(arg, "blosum50") == 0 || strcasecmp(arg, "50") == 0)
        return SCORE_BLOSUM50;
    if (strcasecmp(arg, "blosum62") == 0 || strcasecmp(arg, "62") == 0)
        return SCORE_BLOSUM62;
        
    return SCORE_BLOSUM50;
}

INLINE void init_default_args(void) {
    strcpy(g_args.input_file_path, "./datasets/avpdb.csv");
    strcpy(g_args.output_file_path, "./results/matrix.h5");
    
    g_args.align_method = ALIGN_NEEDLEMAN_WUNSCH;
    g_args.scoring_matrix = SCORE_BLOSUM50;
    g_args.gap_penalty = 4;
    g_args.gap_start = 10;
    g_args.gap_extend = 1;
    g_args.num_threads = 0;
    g_args.compression_level = 0;
    
    g_args.mode_write = 1;
    g_args.mode_benchmark = 0;
    g_args.mode_trim = 0;
    #if MODE_CREATE_ALIGNED_STRINGS == 1
    g_args.aligned_strings = 0;
    #endif
    g_args.verbose = 0;
    g_args.quiet = 0;
}

INLINE void parse_args(int argc, char* argv[]) {
    // (EXPANDABLE)
    int opt;
    int option_index = 0;
    #if MODE_CREATE_ALIGNED_STRINGS == 1
    const char* optstring = "i:o:a:m:p:s:e:t:z:BWTAvqh";
    #else
    const char* optstring = "i:o:a:m:p:s:e:t:z:BWTvqh";
    #endif
    
    while ((opt = getopt_long(argc, argv, optstring, long_options, &option_index)) != -1) {
        switch (opt) {
            case 'i':
                strncpy(g_args.input_file_path, optarg, MAX_PATH - 1);
                g_args.input_file_path[MAX_PATH - 1] = '\0';
                break;
            case 'o':
                strncpy(g_args.output_file_path, optarg, MAX_PATH - 1);
                g_args.output_file_path[MAX_PATH - 1] = '\0';
                break;
            case 'a':
                g_args.align_method = parse_alignment_method(optarg);
                break;
            case 'm':
                g_args.scoring_matrix = parse_scoring_matrix(optarg);
                break;
            case 'p':
                g_args.gap_penalty = atoi(optarg);
                break;
            case 's':
                g_args.gap_start = atoi(optarg);
                break;
            case 'e':
                g_args.gap_extend = atoi(optarg);
                break;
            case 't':
                g_args.num_threads = atoi(optarg);
                if (g_args.num_threads < 0) {
                    g_args.num_threads = 0;
                }
                break;
            case 'z':
                g_args.compression_level = atoi(optarg);
                if (g_args.compression_level < 0 || g_args.compression_level > 9) {
                    g_args.compression_level = 1;
                }
                break;
            case 'B':
                g_args.mode_benchmark = 1;
                break;
            case 'W':
                g_args.mode_write = 0;
                break;
            case 'T':
                g_args.mode_trim = 1;
                break;
            #if MODE_CREATE_ALIGNED_STRINGS == 1
            case 'A':
                g_args.aligned_strings = 1;
                break;
            #endif
            case 'v':
                g_args.verbose = 1;
                break;
            case 'q':
                g_args.quiet = 1;
                break;
            case 'h':
                print_usage(argv[0]);
                exit(0);
            default:
                fprintf(stderr, "Unknown option: %c\n", opt);
                print_usage(argv[0]);
                exit(1);
        }
    }
    
    if (g_args.num_threads == 0) {
        g_args.num_threads = get_thread_count();
    }
}

// forward declaration for get names
INLINE const char* get_alignment_method_name(void);
INLINE const char* get_scoring_matrix_name(void);

INLINE void print_config_section(void) {
    if (g_args.verbose && !g_args.quiet) {
        print_step_header_start("Configuration");
        char buffer[256];
        
        print_config_item("Input", g_args.input_file_path, NULL);
        if (g_args.mode_write) {
            print_config_item("Output", g_args.output_file_path, BOX_TEE_RIGHT);
        } else {
            print_config_item("Output", "Disabled", BOX_TEE_RIGHT);
        }
        print_config_item("Method", get_alignment_method_name(), BOX_TEE_RIGHT);
        print_config_item("Matrix", get_scoring_matrix_name(), BOX_TEE_RIGHT);
        
        if (g_args.align_method == ALIGN_NEEDLEMAN_WUNSCH) {
            snprintf(buffer, sizeof(buffer), "%d", g_args.gap_penalty);
            print_config_item("Gap", buffer, BOX_TEE_RIGHT);
        } else {
            snprintf(buffer, sizeof(buffer), "%d, extend: %d", g_args.gap_start, g_args.gap_extend);
            print_config_item("Gap open", buffer, BOX_TEE_RIGHT);
        }
        
        snprintf(buffer, sizeof(buffer), "%d", g_args.num_threads);
        print_config_item("Threads", buffer, BOX_TEE_RIGHT);
        
        snprintf(buffer, sizeof(buffer), "%d", g_args.compression_level);
        print_config_item("Compression", buffer, BOX_BOTTOM_LEFT);
        
        if (g_args.mode_benchmark) {
            print_timing("Benchmarking mode enabled");
        }
        
        print_step_header_end();
    }
}

INLINE void init_args(int argc, char* argv[]) {
    init_default_args();
    parse_args(argc, argv);
    print_config_section();
}

INLINE const char* get_input_file_path(void) {
    return g_args.input_file_path;
}

INLINE const char* get_output_file_path(void) {
    return g_args.output_file_path;
}

INLINE int get_gap_penalty(void) {
    return g_args.gap_penalty;
}

INLINE int get_gap_start(void) {
    return g_args.gap_start;
}

INLINE int get_gap_extend(void) {
    return g_args.gap_extend;
}

INLINE int get_num_threads(void) {
    return g_args.num_threads;
}

INLINE int get_alignment_method(void) {
    return g_args.align_method;
}

INLINE const char* get_alignment_method_name(void) {
    // (EXPANDABLE)
    return g_args.align_method == ALIGN_NEEDLEMAN_WUNSCH ? "Needleman-Wunsch" :
           g_args.align_method == ALIGN_GOTOH_AFFINE ? "Gotoh (affine)" : "Smith-Waterman";
}

INLINE int get_scoring_matrix(void) {
    return g_args.scoring_matrix;
}

INLINE const char* get_scoring_matrix_name(void) {
    // (EXPANDABLE)
    return g_args.scoring_matrix == SCORE_BLOSUM50 ? "BLOSUM50" : "BLOSUM62";
}

INLINE int get_compression_level(void) {
    return g_args.compression_level;
}

INLINE int get_mode_multithread(void) {
    return g_args.num_threads > 1;
}

INLINE int get_mode_benchmark(void) {
    return g_args.mode_benchmark;
}

INLINE int get_mode_write(void) {
    return g_args.mode_write;
}

INLINE int get_mode_trim(void) {
    return g_args.mode_trim;
}

INLINE int get_aligned_strings(void) {
    #if MODE_CREATE_ALIGNED_STRINGS == 1
    return g_args.aligned_strings;
    #else
    return 0;
    #endif
}

INLINE int get_verbose(void) {
    return g_args.verbose;
}

INLINE int get_quiet(void) {
    return g_args.quiet;
}

#endif // ARGS_H