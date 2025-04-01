#ifndef ARGS_H
#define ARGS_H

#include "arch.h"
#include "files.h"
#include "matrices.h"
#include "methods.h"
#include "print.h"
#include <ctype.h>
#include <getopt.h>

#define PARAM_UNSET -1

typedef struct
{
    char path_input[MAX_PATH];
    char path_output[MAX_PATH];
    int method_id;
    int seq_type;
    int matrix_id;
    int gap_penalty;
    int gap_start;
    int gap_extend;
    int thread_num;
    int compression_level;
    float filter_threshold;

    struct
    {
        unsigned input_file_set : 1;
        unsigned output_file_set : 1;
        unsigned method_id_set : 1;
        unsigned seq_type_set : 1;
        unsigned matrix_set : 1;
        unsigned gap_penalty_set : 1;
        unsigned gap_start_set : 1;
        unsigned gap_extend_set : 1;
        unsigned mode_write : 1;
        unsigned mode_benchmark : 1;
        unsigned mode_filter : 1;
        unsigned verbose : 1;
        unsigned quiet : 1;
    };
} Args;

static Args g_args = { 0 };

static const char* optstring = "i:o:a:t:m:p:s:e:T:z:f:BWlvqh";

static struct option long_options[] = {
    { "input", required_argument, 0, 'i' },     { "output", required_argument, 0, 'o' },
    { "align", required_argument, 0, 'a' },     { "type", required_argument, 0, 't' },
    { "matrix", required_argument, 0, 'm' },    { "gap-penalty", required_argument, 0, 'p' },
    { "gap-start", required_argument, 0, 's' }, { "gap-extend", required_argument, 0, 'e' },
    { "threads", required_argument, 0, 'T' },   { "compression", required_argument, 0, 'z' },
    { "filter", required_argument, 0, 'f' },    { "benchmark", no_argument, 0, 'B' },
    { "no-write", no_argument, 0, 'W' },        { "verbose", no_argument, 0, 'v' },
    { "quiet", no_argument, 0, 'q' },           { "help", no_argument, 0, 'h' },
    { "list-matrices", no_argument, 0, 'l' },   { 0, 0, 0, 0 }
};

#define GETTER(type, name, field)                                                                  \
    INLINE type args_##name(void)                                                                  \
    {                                                                                              \
        return field;                                                                              \
    }

GETTER(const char*, path_input, g_args.path_input)
GETTER(const char*, path_output, g_args.path_output)
GETTER(int, gap_penalty, g_args.gap_penalty)
GETTER(int, gap_start, g_args.gap_start)
GETTER(int, gap_extend, g_args.gap_extend)
GETTER(int, thread_num, g_args.thread_num)
GETTER(int, align_method, g_args.method_id)
GETTER(int, sequence_type, g_args.seq_type)
GETTER(int, scoring_matrix, g_args.matrix_id)
GETTER(int, compression_level, g_args.compression_level)
GETTER(float, filter_threshold, g_args.filter_threshold)
GETTER(bool, mode_multithread, g_args.thread_num > 1)
GETTER(bool, mode_benchmark, g_args.mode_benchmark)
GETTER(bool, mode_filter, g_args.mode_filter)
GETTER(bool, mode_write, g_args.mode_write)

#undef GETTER

INLINE int
args_parse_scoring_matrix(const char* arg, int seq_type)
{
    if (seq_type < 0)
    {
        return PARAM_UNSET;
    }

    if (isdigit(arg[0]))
    {
        int matrix = atoi(arg);
        int max_matrix = (seq_type == SEQ_TYPE_AMINO) ? NUM_AMINO_MATRICES
                                                      : NUM_NUCLEOTIDE_MATRICES;
        if (matrix >= 0 && matrix < max_matrix)
        {
            return matrix;
        }
        return PARAM_UNSET;
    }

    return matrix_name_id(seq_type, arg);
}

INLINE float
args_parse_filter_threshold(const char* arg)
{
    float threshold = atof(arg);
    if (threshold < 0.0f || threshold > 100.0f)
    {
        return -1.0f;
    }
    return threshold > 1.0f ? threshold / 100.0f : threshold;
}

INLINE int
args_parse_thread_num(const char* arg)
{
    int threads = atoi(arg);
    if (threads < 0)
    {
        return 0;
    }
    if (threads > MAX_THREADS)
    {
        return MAX_THREADS;
    }
    return threads;
}

INLINE int
args_parse_compression_level(const char* arg)
{
    int level = atoi(arg);
    return (level < 0 || level > 9) ? 0 : level;
}

INLINE bool
args_validate_file_input(void)
{
    if (!g_args.input_file_set)
    {
        print(ERROR, MSG_NONE, "Missing required parameter: input file (-i, --input)");
        return false;
    }

    FILE* test = fopen(g_args.path_input, "r");
    if (!test)
    {
        print(ERROR, MSG_NONE, "Cannot open input file: %s", g_args.path_input);
        return false;
    }
    fclose(test);
    return true;
}

INLINE bool
args_validate_required(void)
{
    bool valid = true;

    valid &= args_validate_file_input();

    if (!g_args.seq_type_set)
    {
        print(ERROR, MSG_NONE, "Missing required parameter: sequence type (-t, --type)");
        valid = false;
    }

    if (!g_args.method_id_set)
    {
        print(ERROR, MSG_NONE, "Missing required parameter: alignment method (-a, --align)");
        valid = false;
    }

    if (!g_args.matrix_set)
    {
        print(ERROR, MSG_NONE, "Missing required parameter: scoring matrix (-m, --matrix)");
        valid = false;
    }

    if (g_args.method_id_set && g_args.method_id >= 0 && g_args.method_id < ALIGN_COUNT)
    {
        if (methods_alignment_linear(g_args.method_id) && !g_args.gap_penalty_set)
        {
            print(ERROR, MSG_NONE, "Missing required parameter: gap penalty (-p, --gap-penalty)");
            valid = false;
        }
        else if (methods_alignment_affine(g_args.method_id))
        {
            if (!g_args.gap_start_set)
            {
                print(ERROR, MSG_NONE, "Missing required parameter: gap start (-s, --gap-start)");
                valid = false;
            }
            if (!g_args.gap_extend_set)
            {
                print(ERROR, MSG_NONE, "Missing required parameter: gap extend (-e, --gap-extend)");
                valid = false;
            }
        }
    }

    return valid;
}

INLINE void
args_print_matrices(void)
{
    printf("Listing available scoring matrices\n\n");

    printf("Amino Acid Matrices (%d):\n", NUM_AMINO_MATRICES);
    matrix_seq_type_list(SEQ_TYPE_AMINO);
    printf("\n");

    printf("Nucleotide Matrices (%d):\n", NUM_NUCLEOTIDE_MATRICES);
    matrix_seq_type_list(SEQ_TYPE_NUCLEOTIDE);
    printf("\n");
}

INLINE void
args_print_usage(const char* program_name)
{
    printf("Usage: %s [ARGUMENTS]\n\n", program_name);
    printf("Sequence Alignment Tool - Fast pairwise sequence alignment\n\n");

    printf("Required arguments:\n");
    printf("  -i, --input FILE       Input CSV file path\n");
    printf("  -t, --type TYPE        Sequence type\n");

    sequence_types_list();

    printf("  -a, --align METHOD     Alignment method\n");

    methods_alignment_list();

    printf("  -m, --matrix MATRIX    Scoring matrix\n");
    printf("                           Use --list-matrices or -l to see all available matrices\n");
    printf("  -p, --gap-penalty N    Linear gap penalty (for Needleman-Wunsch)\n");
    printf("  -s, --gap-start N      Affine gap start penalty (for affine gap methods)\n");
    printf("  -e, --gap-extend N     Affine gap extend penalty (for affine gap methods)\n");

    printf("\nOptional arguments:\n");
    printf("  -o, --output FILE      Output HDF5 file path (required for writing results)\n");
    printf("  -T, --threads N        Number of threads (0 = auto)\n");
    printf("  -z, --compression N    HDF5 compression level (0-9) [default: 0 (no compression)]\n");
    printf("  -f, --filter THRESHOLD Filter sequences with similarity above threshold\n");
    printf("  -B, --benchmark        Enable benchmarking mode\n");
    printf("  -W, --no-write         Disable writing to output file\n");
    printf("  -v, --verbose          Enable verbose output\n");
    printf("  -q, --quiet            Suppress all non-error output\n");
    printf("  -l, --list-matrices    List all available scoring matrices\n");
    printf("  -h, --help             Display this help message\n");
}

INLINE void
args_print_config(void)
{
    if (g_args.quiet)
    {
        return;
    }

    print(SECTION, MSG_NONE, "Configuration");

    print(CONFIG, MSG_LOC(FIRST), "Input: %s", file_name_path(g_args.path_input));

    if (g_args.output_file_set)
    {
        print(CONFIG, MSG_LOC(MIDDLE), "Output: %s", file_name_path(g_args.path_output));
    }
    else
    {
        print(CONFIG, MSG_LOC(MIDDLE), "Output: Disabled (no output file specified)");
    }

    print(CONFIG, MSG_LOC(MIDDLE), "Method: %s", methods_alignment_name(g_args.method_id));
    print(CONFIG, MSG_LOC(MIDDLE), "Sequence type: %s", sequence_type_name(g_args.seq_type));
    print(CONFIG, MSG_LOC(MIDDLE), "Matrix: %s", matrix_id_name(g_args.seq_type, g_args.matrix_id));

    if (methods_alignment_linear(g_args.method_id) && g_args.gap_penalty_set)
    {
        print(CONFIG, MSG_LOC(MIDDLE), "Gap: %d", g_args.gap_penalty);
    }
    else if (methods_alignment_affine(g_args.method_id) &&
             (g_args.gap_start_set && g_args.gap_extend_set))
    {
        print(CONFIG,
              MSG_LOC(MIDDLE),
              "Gap open: %d, extend: %d",
              g_args.gap_start,
              g_args.gap_extend);
    }

    if (g_args.mode_filter)
    {
        print(CONFIG,
              MSG_LOC(MIDDLE),
              "Filter threshold: %.1f%%",
              g_args.filter_threshold * 100.0f);
    }

    if (g_args.mode_write)
    {
        print(CONFIG, MSG_LOC(MIDDLE), "Compression: %d", g_args.compression_level);
    }

    print(CONFIG, MSG_LOC(LAST), "Threads: %d", g_args.thread_num);

    if (g_args.mode_benchmark)
    {
        print(TIMING, MSG_NONE, "Benchmarking mode enabled");
    }
}

INLINE void
args_parse(int argc, char* argv[])
{
    int opt;
    int option_index = 0;

    while ((opt = getopt_long(argc, argv, optstring, long_options, &option_index)) != -1)
    {
        switch (opt)
        {
            case 'i':
                strncpy(g_args.path_input, optarg, MAX_PATH - 1);
                g_args.input_file_set = 1;
                break;

            case 'o':
                strncpy(g_args.path_output, optarg, MAX_PATH - 1);
                g_args.output_file_set = 1;
                g_args.mode_write = 1;
                break;

            case 'a':
                g_args.method_id = methods_alignment_arg(optarg);
                if (g_args.method_id != PARAM_UNSET)
                {
                    g_args.method_id_set = 1;
                }
                else
                {
                    print(ERROR, MSG_NONE, "Unknown alignment method: %s", optarg);
                }
                break;

            case 't':
                g_args.seq_type = sequence_type_arg(optarg);
                if (g_args.seq_type != PARAM_UNSET)
                {
                    g_args.seq_type_set = 1;
                }
                else
                {
                    print(ERROR, MSG_NONE, "Unknown sequence type: %s", optarg);
                }
                break;

            case 'm':
                if (!g_args.seq_type_set)
                {
                    print(ERROR, MSG_NONE, "Must specify sequence type (-t) before matrix");
                    break;
                }
                g_args.matrix_id = args_parse_scoring_matrix(optarg, g_args.seq_type);
                if (g_args.matrix_id != PARAM_UNSET)
                {
                    g_args.matrix_set = 1;
                }
                else
                {
                    print(ERROR, MSG_NONE, "Unknown scoring matrix: %s", optarg);
                }
                break;

            case 'p':
                g_args.gap_penalty = atoi(optarg);
                g_args.gap_penalty_set = 1;
                break;

            case 's':
                g_args.gap_start = atoi(optarg);
                g_args.gap_start_set = 1;
                break;

            case 'e':
                g_args.gap_extend = atoi(optarg);
                g_args.gap_extend_set = 1;
                break;

            case 'T':
                g_args.thread_num = args_parse_thread_num(optarg);
                break;

            case 'z':
                g_args.compression_level = args_parse_compression_level(optarg);
                break;

            case 'f':
                g_args.filter_threshold = args_parse_filter_threshold(optarg);
                if (g_args.filter_threshold >= 0)
                {
                    g_args.mode_filter = 1;
                }
                else
                {
                    print(ERROR, MSG_NONE, "Invalid filter threshold: %s", optarg);
                }
                break;

            case 'B':
                g_args.mode_benchmark = 1;
                break;

            case 'W':
                g_args.mode_write = 0;
                break;

            case 'v':
                g_args.verbose = 1;
                print_verbose_flip();
                break;

            case 'q':
                g_args.quiet = 1;
                print_quiet_flip();
                break;

            case 'l':
                args_print_matrices();
                exit(0);

            case 'h':
                args_print_usage(argv[0]);
                exit(0);

            default:
                print(ERROR, MSG_NONE, "Unknown option: %c", opt);
                args_print_usage(argv[0]);
                exit(1);
        }
    }

    if (g_args.thread_num == 0)
    {
        g_args.thread_num = thread_count();
    }
}

INLINE void
args_init(int argc, char* argv[])
{
    g_args.method_id = PARAM_UNSET;
    g_args.seq_type = PARAM_UNSET;
    g_args.matrix_id = PARAM_UNSET;

    args_parse(argc, argv);
    print_context_init();

    if (!args_validate_required())
    {
        print(SECTION, MSG_NONE, NULL);
        printf("\nPlease check the usage below\n\n");
        args_print_usage(argv[0]);
        exit(1);
    }
}

#endif // ARGS_H