#ifndef ARGS_H
#define ARGS_H

#include "common.h"
#include "matrices.h"
#include "methods.h"
#include "print.h"

#define PARAM_UNSET -1

typedef struct
{
    char input_file_path[MAX_PATH];
    char output_file_path[MAX_PATH];
    int align_method;
    int seq_type;
    int scoring_matrix;
    int gap_penalty;
    int gap_start;
    int gap_extend;
    int num_threads;
    int compression_level;
    float filter_threshold;

    unsigned mode_benchmark : 1;
    unsigned mode_write : 1;
    unsigned mode_filter : 1;
    unsigned verbose : 1;
    unsigned quiet : 1;

    unsigned input_file_set : 1;
    unsigned output_file_set : 1;
    unsigned align_method_set : 1;
    unsigned seq_type_set : 1;
    unsigned matrix_set : 1;
    unsigned gap_penalty_set : 1;
    unsigned gap_start_set : 1;
    unsigned gap_extend_set : 1;
    unsigned no_write_flag_set : 1;
} Args;

// Long options for getopt
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

static Args g_args = { 0 };

// Accessor functions for configuration values
INLINE const char*
get_input_file_path(void)
{
    return g_args.input_file_path;
}

INLINE const char*
get_output_file_path(void)
{
    return g_args.output_file_path;
}

INLINE int
get_gap_penalty(void)
{
    return g_args.gap_penalty;
}

INLINE int
get_gap_start(void)
{
    return g_args.gap_start;
}

INLINE int
get_gap_extend(void)
{
    return g_args.gap_extend;
}

INLINE int
get_num_threads(void)
{
    return g_args.num_threads;
}

INLINE int
get_alignment_method(void)
{
    return g_args.align_method;
}

INLINE int
get_sequence_type(void)
{
    return g_args.seq_type;
}

INLINE int
get_scoring_matrix(void)
{
    return g_args.scoring_matrix;
}

INLINE int
get_compression_level(void)
{
    return g_args.compression_level;
}

INLINE bool
get_mode_multithread(void)
{
    return g_args.num_threads > 1;
}

INLINE bool
get_mode_benchmark(void)
{
    return g_args.mode_benchmark;
}

INLINE bool
get_mode_write(void)
{
    return g_args.mode_write;
}

INLINE bool
get_mode_filter(void)
{
    return g_args.mode_filter;
}

INLINE float
get_filter_threshold(void)
{
    return g_args.filter_threshold;
}

INLINE bool
get_verbose(void)
{
    return g_args.verbose;
}

INLINE bool
get_quiet(void)
{
    return g_args.quiet;
}

INLINE const char*
get_current_alignment_method_name(void)
{
    return get_alignment_method_name(g_args.align_method);
}

INLINE const char*
get_sequence_type_name(void)
{
    if (g_args.seq_type >= 0 && g_args.seq_type < SEQ_TYPE_COUNT)
    {
        return SEQUENCE_TYPES[g_args.seq_type].name;
    }

    return "Unknown";
}

INLINE const char*
get_scoring_matrix_name(void)
{
    return get_matrix_name_by_id(g_args.seq_type, g_args.scoring_matrix);
}

INLINE void
print_available_matrices(void)
{
    printf("Listing available scoring matrices\n\n");

    printf("Amino Acid Matrices (%d):\n", NUM_AMINO_MATRICES);
    list_matrices_for_seq_type(SEQ_TYPE_AMINO);
    printf("\n");

    printf("Nucleotide Matrices (%d):\n", NUM_NUCLEOTIDE_MATRICES);
    list_matrices_for_seq_type(SEQ_TYPE_NUCLEOTIDE);
    printf("\n");
}

INLINE void
print_usage(const char* program_name)
{
    printf("Usage: %s [ARGUMENTS]\n\n", program_name);
    printf("Sequence Alignment Tool - Fast pairwise sequence alignment\n\n");

    printf("Required arguments:\n");
    printf("  -i, --input FILE       Input CSV file path\n");
    printf("  -t, --type TYPE        Sequence type\n");

    for (int i = 0; i < SEQ_TYPE_COUNT; i++)
    {
        printf("                           %s: %s (%s)\n",
               SEQUENCE_TYPES[i].aliases[0],
               SEQUENCE_TYPES[i].name,
               SEQUENCE_TYPES[i].description);
    }

    printf("  -a, --align METHOD     Alignment method\n");

    for (int i = 0; i < ALIGN_COUNT; i++)
    {
        printf("                           %s: %s (%s, %s gap)\n",
               ALIGNMENT_METHODS[i].aliases[0],
               ALIGNMENT_METHODS[i].name,
               ALIGNMENT_METHODS[i].description,
               get_gap_type_name(i));
    }

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
print_config_section(void)
{
    if (g_args.quiet)
    {
        return;
    }

    print_step_header_start("Configuration");
    char buffer[256];

    print_config_item("Input", get_file_name(g_args.input_file_path), NULL);

    if (g_args.output_file_set)
    {
        print_config_item("Output", get_file_name(g_args.output_file_path), BOX_TEE_RIGHT);
    }

    else if (g_args.no_write_flag_set)
    {
        print_config_item("Output", "Disabled (--no-write)", BOX_TEE_RIGHT);
    }

    else
    {
        print_config_item("Output", "Disabled (no output file specified)", BOX_TEE_RIGHT);
    }

    print_config_item("Method", get_current_alignment_method_name(), BOX_TEE_RIGHT);
    print_config_item("Sequence type", get_sequence_type_name(), BOX_TEE_RIGHT);
    print_config_item("Matrix", get_scoring_matrix_name(), BOX_TEE_RIGHT);

    if (g_args.align_method_set && g_args.align_method >= 0 && g_args.align_method < ALIGN_COUNT)
    {
        GapPenaltyType gap_type = ALIGNMENT_METHODS[g_args.align_method].gap_type;

        if (gap_type == GAP_TYPE_LINEAR && g_args.gap_penalty_set)
        {
            snprintf(buffer, sizeof(buffer), "%d", g_args.gap_penalty);
            print_config_item("Gap", buffer, BOX_TEE_RIGHT);
        }

        else if (gap_type == GAP_TYPE_AFFINE && (g_args.gap_start_set && g_args.gap_extend_set))
        {
            snprintf(buffer, sizeof(buffer), "%d, extend: %d", g_args.gap_start, g_args.gap_extend);
            print_config_item("Gap open", buffer, BOX_TEE_RIGHT);
        }
    }

    snprintf(buffer, sizeof(buffer), "%d", g_args.num_threads);
    print_config_item("Threads", buffer, BOX_TEE_RIGHT);

    if (g_args.mode_filter)
    {
        snprintf(buffer, sizeof(buffer), "%.1f%%", g_args.filter_threshold * 100.0f);
        print_config_item("Filter threshold", buffer, BOX_TEE_RIGHT);
    }

    if (g_args.mode_write)
    {
        snprintf(buffer, sizeof(buffer), "%d", g_args.compression_level);
        print_config_item("Compression", buffer, BOX_BOTTOM_LEFT);
    }

    if (g_args.mode_benchmark)
    {
        print_timing("Benchmarking mode enabled");
    }

    print_step_header_end(0);
}

// Validation functions
INLINE bool
validate_input_file(void)
{
    if (!g_args.input_file_set)
    {
        print_error("Missing required parameter: input file (-i, --input)\n");
        return false;
    }

    FILE* test = fopen(g_args.input_file_path, "r");
    if (!test)
    {
        print_error("Cannot open input file: %s\n", g_args.input_file_path);
        return false;
    }

    fclose(test);
    return true;
}

INLINE bool
validate_sequence_type(void)
{
    if (!g_args.seq_type_set)
    {
        print_error("Missing required parameter: sequence type (-t, --type)\n");
        return false;
    }

    return true;
}

INLINE bool
validate_alignment_method(void)
{
    if (!g_args.align_method_set)
    {
        print_error("Missing required parameter: alignment method (-a, --align)\n");
        return false;
    }

    return true;
}

INLINE bool
validate_scoring_matrix(void)
{
    if (!g_args.matrix_set)
    {
        print_error("Missing required parameter: scoring matrix (-m, --matrix)\n");
        return false;
    }

    return true;
}

INLINE bool
validate_gap_penalties(void)
{
    if (!g_args.align_method_set || g_args.align_method < 0 || g_args.align_method >= ALIGN_COUNT)
    {
        return false;
    }

    GapPenaltyType gap_type = ALIGNMENT_METHODS[g_args.align_method].gap_type;

    if (gap_type == GAP_TYPE_LINEAR && !g_args.gap_penalty_set)
    {
        print_error("Missing required parameter: gap penalty (-p, --gap-penalty) for %s\n",
                    get_alignment_method_name(g_args.align_method));

        return false;
    }

    if (gap_type == GAP_TYPE_AFFINE)
    {
        if (!g_args.gap_start_set)
        {
            print_error("Missing required parameter: gap start (-s, --gap-start) for %s\n",
                        get_alignment_method_name(g_args.align_method));

            return false;
        }

        if (!g_args.gap_extend_set)
        {
            print_error("Missing required parameter: gap extend (-e, --gap-extend) for %s\n",
                        get_alignment_method_name(g_args.align_method));

            return false;
        }
    }

    return true;
}

INLINE bool
validate_required_arguments(void)
{
    bool valid = true;

    valid &= validate_input_file();
    valid &= validate_sequence_type();
    valid &= validate_alignment_method();
    valid &= validate_scoring_matrix();
    valid &= validate_gap_penalties();

    return valid;
}

// Parsing functions
INLINE int
parse_sequence_type(const char* arg)
{
    if (!arg)
    {
        return PARAM_UNSET;
    }

    if (isdigit(arg[0]) || (arg[0] == '-' && isdigit(arg[1])))
    {
        int type = atoi(arg);
        if (type >= 0 && type < SEQ_TYPE_COUNT)
        {
            return type;
        }

        print_error("Invalid sequence type index %d, valid range is 0-%d",
                    type,
                    SEQ_TYPE_COUNT - 1);

        return PARAM_UNSET;
    }

    for (int i = 0; i < SEQ_TYPE_COUNT; i++)
    {
        for (const char** alias = SEQUENCE_TYPES[i].aliases; *alias != NULL; alias++)
        {
            if (strcasecmp(arg, *alias) == 0)
            {
                return SEQUENCE_TYPES[i].type;
            }
        }
    }

    print_error("Unknown sequence type '%s'", arg);
    return PARAM_UNSET;
}

INLINE int
parse_scoring_matrix(const char* arg, int seq_type)
{
    if (!arg)
    {
        return PARAM_UNSET;
    }

    if (seq_type < 0 || seq_type >= SEQ_TYPE_COUNT)
    {
        print_error("Cannot select scoring matrix without specifying sequence type first");
        return PARAM_UNSET;
    }

    if (isdigit(arg[0]) || (arg[0] == '-' && isdigit(arg[1])))
    {
        int matrix_id = atoi(arg);
        int max_matrix = seq_type == SEQ_TYPE_AMINO ? NUM_AMINO_MATRICES - 1
                                                    : NUM_NUCLEOTIDE_MATRICES - 1;

        if (matrix_id >= 0 && matrix_id <= max_matrix)
        {
            return matrix_id;
        }

        else
        {
            print_error("Invalid matrix index %d, valid range is 0-%d", matrix_id, max_matrix);
            return PARAM_UNSET;
        }
    }

    int matrix_id = find_matrix_by_name(seq_type, arg);
    if (matrix_id < 0)
    {
        print_error("Unknown scoring matrix '%s' for %s sequences",
                    arg,
                    seq_type == SEQ_TYPE_AMINO ? "amino acid" : "nucleotide");

        return PARAM_UNSET;
    }

    return matrix_id;
}

INLINE float
parse_filter_threshold(const char* arg)
{
    float value = atof(arg);

    if (value < 0.0f || value > 100.0f)
    {
        print_error("Invalid filter threshold %.2f, it must be a percentage between 0 and 100 or a "
                    "decimal between 0 and 1\n",
                    value);

        print_step_header_end(1);
        exit(1);
    }

    if (value > 1.0f)
    {
        value /= 100.0f;
    }

    return value;
}

INLINE int
parse_thread_count(const char* arg)
{
    int num_threads = atoi(arg);

    if (num_threads < 0)
    {
        num_threads = 0;
    }

    else if (num_threads > MAX_THREADS)
    {
        print_warning("Exceeded maximum thread count, using %d", MAX_THREADS);
        num_threads = MAX_THREADS;
    }

    else if (num_threads == get_thread_count())
    {
        print_warning("Invalid thread count, using auto-detected value");
        num_threads = 0;
    }

    return num_threads;
}

INLINE int
parse_compression_level(const char* arg)
{
    int level = atoi(arg);

    if (level < 0 || level > 9)
    {
        print_error("Invalid compression level %d, valid range is 0-9\n", level);
        print_step_header_end(1);
        exit(1);
    }

    return level;
}

INLINE void
handle_alignment_method_arg(const char* arg)
{
    int method = find_alignment_method_by_name(arg);
    if (method != -1)
    {
        g_args.align_method = method;
        g_args.align_method_set = 1;
    }
    else
    {
        print_error("Unknown alignment method '%s'", arg);
    }
}

INLINE void
handle_sequence_type_arg(const char* arg)
{
    int type = parse_sequence_type(arg);
    if (type != PARAM_UNSET)
    {
        g_args.seq_type = type;
        g_args.seq_type_set = 1;
    }
}

INLINE void
handle_matrix_arg(const char* arg)
{
    if (!g_args.seq_type_set)
    {
        print_error("Must specify sequence type (-t) before specifying matrix\n");
    }

    else
    {
        int matrix_id = parse_scoring_matrix(arg, g_args.seq_type);
        if (matrix_id != PARAM_UNSET)
        {
            g_args.scoring_matrix = matrix_id;
            g_args.matrix_set = 1;
        }
    }
}

INLINE void
parse_args(int argc, char* argv[])
{
    int opt;
    int option_index = 0;
    const char* optstring = "i:o:a:t:m:p:s:e:T:z:f:BWlvqh";

    while ((opt = getopt_long(argc, argv, optstring, long_options, &option_index)) != -1)
    {
        switch (opt)
        {
            case 'i':
                strncpy(g_args.input_file_path, optarg, MAX_PATH - 1);
                g_args.input_file_path[MAX_PATH - 1] = '\0';
                g_args.input_file_set = 1;
                break;

            case 'o':
                strncpy(g_args.output_file_path, optarg, MAX_PATH - 1);
                g_args.output_file_path[MAX_PATH - 1] = '\0';
                g_args.output_file_set = 1;
                g_args.mode_write = 1;
                break;

            case 'a':
                handle_alignment_method_arg(optarg);
                break;

            case 't':
                handle_sequence_type_arg(optarg);
                break;

            case 'm':
                handle_matrix_arg(optarg);
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
                g_args.num_threads = parse_thread_count(optarg);
                break;

            case 'z':
                g_args.compression_level = parse_compression_level(optarg);
                break;

            case 'f':
                g_args.filter_threshold = parse_filter_threshold(optarg);
                g_args.mode_filter = 1;
                break;

            case 'B':
                g_args.mode_benchmark = 1;
                break;

            case 'W':
                g_args.mode_write = 0;
                g_args.no_write_flag_set = 1;
                break;

            case 'v':
                g_args.verbose = 1;
                break;

            case 'q':
                g_args.quiet = 1;
                break;

            case 'l':
                print_available_matrices();
                exit(0);

            case 'h':
                print_usage(argv[0]);
                exit(0);

            default:
                print_error("Unknown option: %c\n", opt);
                print_step_header_end(1);
                print_usage(argv[0]);
                exit(1);
        }
    }

    if (g_args.num_threads == 0)
    {
        g_args.num_threads = get_thread_count();
    }
}

INLINE void
finalize_args_initialization(int argc, char* argv[])
{
    if (!validate_required_arguments())
    {
        print_step_header_end(1);
        printf("\nPlease check the usage below to properly start the program\n\n");
        print_usage(argv[0]);
        exit(1);
    }

    print_config_section();
}

INLINE void
init_args(int argc, char* argv[])
{
    parse_args(argc, argv);
    init_print_messages(g_args.verbose, g_args.quiet);
    finalize_args_initialization(argc, argv);
}

#endif // ARGS_H