#ifndef ARGS_H
#define ARGS_H

#include "common.h"
#include <ctype.h>
#include <getopt.h>

// Alignment methods
#define ALIGN_NEEDLEMAN_WUNSCH 0
#define ALIGN_GOTOH_AFFINE     1
#define ALIGN_SMITH_WATERMAN   2

// Scoring matrices
#define SCORE_BLOSUM50 0
#define SCORE_BLOSUM62 1

typedef struct {
    char input_file_path[MAX_PATH];
    char output_file_path[MAX_PATH];
    int align_method;           // 0=NW, 1=GA, 2=SW
    int scoring_matrix;         // 0=BLOSUM50, 1=BLOSUM62
    int gap_penalty;            // Linear gap penalty (NW)
    int gap_start;              // Affine gap start penalty (GA|SW)
    int gap_extend;             // Affine gap extend penalty (GA|SW)

    // TODO: Move to skip_header() in csv.h
    int seq_col;                // CSV sequence column (0-based)
    int num_cols;               // Number of columns in CSV

    int num_threads;            // Number of threads (0 = auto)
    int compression_level;      // HDF5 compression level (0-9)
    
    // Flags
    unsigned mode_benchmark : 1; // Enable benchmarking
    unsigned mode_write : 1;     // Enable output file writing
    unsigned mode_trim : 1;      // Enable sequence trimming
    unsigned aligned_strings : 1;// Create strings with gaps vs score-only
    unsigned verbose : 1;        // Verbose output
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
    {"seq-column",      required_argument, 0, 'c'},
    {"num-columns",     required_argument, 0, 'n'},
    {"threads",         required_argument, 0, 't'},
    {"compression",     required_argument, 0, 'z'},
    {"benchmark",       no_argument,       0, 'B'},
    {"no-write",        no_argument,       0, 'W'},
    {"trim",            no_argument,       0, 'T'},
    {"aligned-strings", no_argument,       0, 'A'},
    {"verbose",         no_argument,       0, 'v'},
    {"help",            no_argument,       0, 'h'},
    {0, 0, 0, 0}
};

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
    printf("  -c, --seq-column N     CSV sequence column (0-based) [default: 0]\n");
    printf("  -n, --num-columns N    Number of columns in CSV [default: 2]\n");
    printf("  -t, --threads N        Number of threads (0 = auto) [default: 0]\n");
    printf("  -z, --compression N    HDF5 compression level (0-9) [default: 1]\n");
    printf("  -B, --benchmark        Enable benchmarking mode\n");
    printf("  -W, --no-write         Disable writing to output file\n");
    printf("  -T, --trim             Enable sequence trimming\n");
    printf("  -A, --aligned-strings  Create aligned strings with gaps (slower)\n");
    printf("  -v, --verbose          Enable verbose output\n");
    printf("  -h, --help             Display this help message\n");
}

INLINE int parse_alignment_method(const char* arg) {
    if (!arg) return ALIGN_NEEDLEMAN_WUNSCH;
    
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

INLINE void init_default_args(Args* args) {
    memset(args, 0, sizeof(Args));
    
    strcpy(args->input_file_path, "./datasets/avpdb.csv");
    strcpy(args->output_file_path, "./results/matrix.h5");
    
    args->align_method = ALIGN_NEEDLEMAN_WUNSCH;
    args->scoring_matrix = SCORE_BLOSUM50;
    args->gap_penalty = 4;
    args->gap_start = 10;
    args->gap_extend = 1;
    args->seq_col = 0;
    args->num_cols = 2;
    args->num_threads = 0;
    args->compression_level = 0;
    
    args->mode_write = 1;
    args->mode_benchmark = 0;
    args->mode_trim = 0;
    args->aligned_strings = 0;
    args->verbose = 0;
}

INLINE Args* parse_args(int argc, char* argv[]) {
    static Args args;
    init_default_args(&args);
    
    int opt;
    int option_index = 0;
    const char* optstring = "i:o:a:m:p:s:e:c:n:t:z:BWTAvh";
    
    while ((opt = getopt_long(argc, argv, optstring, long_options, &option_index)) != -1) {
        switch (opt) {
            case 'i':
                strncpy(args.input_file_path, optarg, MAX_PATH - 1);
                args.input_file_path[MAX_PATH - 1] = '\0';
                break;
            case 'o':
                strncpy(args.output_file_path, optarg, MAX_PATH - 1);
                args.output_file_path[MAX_PATH - 1] = '\0';
                break;
            case 'a':
                args.align_method = parse_alignment_method(optarg);
                break;
            case 'm':
                args.scoring_matrix = parse_scoring_matrix(optarg);
                break;
            case 'p':
                args.gap_penalty = atoi(optarg);
                break;
            case 's':
                args.gap_start = atoi(optarg);
                break;
            case 'e':
                args.gap_extend = atoi(optarg);
                break;
            case 'c':
                args.seq_col = atoi(optarg);
                break;
            case 'n':
                args.num_cols = atoi(optarg);
                break;
            case 't':
                args.num_threads = atoi(optarg);
                if (args.num_threads < 0) {
                    args.num_threads = 0;
                }
                break;
            case 'z':
                args.compression_level = atoi(optarg);
                if (args.compression_level < 0 || args.compression_level > 9) {
                    args.compression_level = 1;
                }
                break;
            case 'B':
                args.mode_benchmark = 1;
                break;
            case 'W':
                args.mode_write = 0;
                break;
            case 'T':
                args.mode_trim = 1;
                break;
            case 'A':
                args.aligned_strings = 1;
                break;
            case 'v':
                args.verbose = 1;
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
    
    if (args.num_threads == 0) {
        args.num_threads = get_thread_count();
    }
    
    if (args.mode_benchmark && args.verbose) {
        printf("Note: Benchmarking mode enabled. Timing information will be displayed.\n");
    }
    
    return &args;
}

static Args* g_args = NULL;

INLINE void init_args(int argc, char* argv[]) {
    g_args = parse_args(argc, argv);
    
    if (g_args->verbose) {
        printf("Configuration:\n");
        printf("  Input file: %s\n", g_args->input_file_path);
        printf("  Output file: %s\n", g_args->output_file_path);
        printf("  Alignment method: %s\n", 
            g_args->align_method == ALIGN_NEEDLEMAN_WUNSCH ? "Needleman-Wunsch" :
            g_args->align_method == ALIGN_GOTOH_AFFINE ? "Gotoh (affine)" : "Smith-Waterman");
        printf("  Gap settings: ");
        if (g_args->align_method == ALIGN_NEEDLEMAN_WUNSCH) {
            printf("Linear penalty = %d\n", g_args->gap_penalty);
        } else {
            printf("Affine (start = %d, extend = %d)\n", g_args->gap_start, g_args->gap_extend);
        }
        printf("  Threads: %d\n", g_args->num_threads);
    }
}

INLINE const char* get_input_file_path(void) {
    return g_args->input_file_path;
}

INLINE const char* get_output_file_path(void) {
    return g_args->output_file_path;
}

INLINE int get_gap_penalty(void) {
    return g_args->gap_penalty;
}

INLINE int get_gap_start(void) {
    return g_args->gap_start;
}

INLINE int get_gap_extend(void) {
    return g_args->gap_extend;
}

INLINE int get_seq_col(void) {
    return g_args->seq_col;
}

INLINE int get_num_cols(void) {
    return g_args->num_cols;
}

INLINE int get_num_threads(void) {
    return g_args->num_threads;
}

INLINE int get_alignment_method(void) {
    return g_args->align_method;
}

INLINE int get_scoring_matrix(void) {
    return g_args->scoring_matrix;
}

INLINE int get_compression_level(void) {
    return g_args->compression_level;
}

INLINE int get_mode_multithread(void) {
    return g_args->num_threads > 1;
}

INLINE int get_mode_benchmark(void) {
    return g_args->mode_benchmark;
}

INLINE int get_mode_write(void) {
    return g_args->mode_write;
}

INLINE int get_mode_trim(void) {
    return g_args->mode_trim;
}

INLINE int get_aligned_strings(void) {
    return g_args->aligned_strings;
}

INLINE int get_verbose(void) {
    return g_args->verbose;
}

#endif // ARGS_H