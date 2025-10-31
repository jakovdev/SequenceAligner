#include "core/app/args.h"

#include <ctype.h>
#include <getopt.h>
#include <limits.h>
#include <omp.h>
#include <stdio.h>

#include "core/bio/score/matrices.h"
#include "core/bio/types.h"
#include "system/arch.h"
#include "util/print.h"

#define PARAM_UNSET -1

static struct
{
    char path_input[MAX_PATH];
    char path_output[MAX_PATH];
    SequenceType seq_type;
    int matrix_id;
    AlignmentMethod method_id;
    int gap_penalty;
    int gap_open;
    int gap_extend;
    float filter;
    unsigned long thread_num;
    unsigned int compression_level;

    unsigned input_file_set : 1;
    unsigned output_file_set : 1;
    unsigned seq_type_set : 1;
    unsigned matrix_set : 1;
    unsigned method_id_set : 1;
    unsigned gap_penalty_set : 1;
    unsigned gap_open_set : 1;
    unsigned gap_extend_set : 1;
    unsigned mode_write : 1;
    unsigned mode_filter : 1;
    unsigned mode_benchmark : 1;
    unsigned mode_cuda : 1;
    unsigned quiet : 1;
} args = { 0 };

static const char* optstring = "i:t:m:a:p:s:e:o:f:T:z:BCWDvqlh";

static struct option long_options[] = { { "input", required_argument, 0, 'i' },
                                        { "type", required_argument, 0, 't' },
                                        { "matrix", required_argument, 0, 'm' },
                                        { "align", required_argument, 0, 'a' },
                                        { "gap-penalty", required_argument, 0, 'p' },
                                        { "gap-open", required_argument, 0, 's' },
                                        { "gap-extend", required_argument, 0, 'e' },
                                        { "output", required_argument, 0, 'o' },
                                        { "filter", required_argument, 0, 'f' },
                                        { "threads", required_argument, 0, 'T' },
                                        { "compression", required_argument, 0, 'z' },
                                        { "benchmark", no_argument, 0, 'B' },
                                        { "no-cuda", no_argument, 0, 'C' },
                                        { "no-write", no_argument, 0, 'W' },
                                        { "no-detail", no_argument, 0, 'D' },
                                        { "verbose", no_argument, 0, 'v' },
                                        { "quiet", no_argument, 0, 'q' },
                                        { "list-matrices", no_argument, 0, 'l' },
                                        { "help", no_argument, 0, 'h' },
                                        { 0, 0, 0, 0 } };

#define GETTER(type, name, field)                                                                  \
    type args_##name(void)                                                                         \
    {                                                                                              \
        return field;                                                                              \
    }

GETTER(const char*, input, args.path_input)
GETTER(const char*, output, args.path_output)
GETTER(int, gap_penalty, args.gap_penalty)
GETTER(int, gap_open, args.gap_open)
GETTER(int, gap_extend, args.gap_extend)
GETTER(unsigned long, thread_num, args.thread_num)
GETTER(AlignmentMethod, align_method, args.method_id)
GETTER(SequenceType, sequence_type, args.seq_type)
GETTER(int, scoring_matrix, args.matrix_id)
GETTER(unsigned int, compression, args.compression_level)
GETTER(float, filter, args.filter)
GETTER(bool, mode_benchmark, (bool)args.mode_benchmark)
GETTER(bool, mode_write, (bool)args.mode_write)
GETTER(bool, mode_cuda, (bool)args.mode_cuda)

#undef GETTER

static bool
args_validate_file_input(void)
{
    if (!args.input_file_set)
    {
        print(ERROR, MSG_NONE, "Missing parameter: input file (-i, --input)");
        return false;
    }

    FILE* test = fopen(args.path_input, "r");
    if (!test)
    {
        print(ERROR, MSG_NONE, "Cannot open input file: %s", args.path_input);
        return false;
    }

    fclose(test);
    return true;
}

static bool
args_validate_required(void)
{
    bool valid = true;

    valid &= args_validate_file_input();

    if (!args.seq_type_set)
    {
        print(ERROR, MSG_NONE, "Missing parameter: sequence type (-t, --type)");
        valid = false;
    }

    if (!args.matrix_set)
    {
        print(ERROR, MSG_NONE, "Missing parameter: substitution matrix (-m, --matrix)");
        valid = false;
    }

    if (!args.method_id_set)
    {
        print(ERROR, MSG_NONE, "Missing parameter: alignment method (-a, --align)");
        valid = false;
    }

    if (args.method_id_set && args.method_id >= 0 && args.method_id < ALIGN_COUNT)
    {
        if (alignment_linear(args.method_id) && !args.gap_penalty_set)
        {
            print(ERROR, MSG_NONE, "Missing parameter: gap penalty (-p, --gap-penalty)");
            valid = false;
        }

        else if (alignment_affine(args.method_id))
        {
            if (!args.gap_open_set)
            {
                print(ERROR, MSG_NONE, "Missing parameter: gap open (-s, --gap-open)");
                valid = false;
            }

            if (!args.gap_extend_set)
            {
                print(ERROR, MSG_NONE, "Missing parameter: gap extend (-e, --gap-extend)");
                valid = false;
            }
        }
    }

    return valid;
}

static void
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

static void
args_print_usage(const char* program_name)
{
    printf("Usage: %s [ARGUMENTS]\n\n", program_name);
    printf("Sequence Alignment Tool - Fast pairwise sequence alignment\n\n");

    printf("Required arguments:\n");
    printf("  -i, --input FILE       Input CSV file path\n");

    printf("  -t, --type TYPE        Sequence type\n");
    sequence_types_list();

    printf("  -m, --matrix MATRIX    Scoring matrix\n");
    printf("                           Use --list-matrices or -l to see all available matrices\n");

    printf("  -a, --align METHOD     Alignment method\n");
    alignment_list();

    printf("  -p, --gap-penalty N    Linear gap penalty (for Needleman-Wunsch)\n");
    printf("  -s, --gap-open N       Affine gap open penalty (for affine gap methods)\n");
    printf("  -e, --gap-extend N     Affine gap extend penalty (for affine gap methods)\n");

    printf("\nOptional arguments:\n");
    printf("  -o, --output FILE      Output HDF5 file path (required for writing results)\n");
    printf("  -f, --filter THRESHOLD Filter sequences with similarity above threshold (0.0-1.0)\n");
    printf("  -T, --threads N        Number of threads (0 = auto)\n");
    printf("  -z, --compression N    HDF5 compression level (0-9) [default: 0 (no compression)]\n");
    printf("  -B, --benchmark        Enable benchmarking mode\n");
#ifdef USE_CUDA
    printf("  -C, --no-cuda          Disable CUDA support\n");
#endif
    printf("  -W, --no-write         Disable writing to output file\n");
    printf("  -D, --no-detail        Disable detailed printing\n");
    printf("  -v, --verbose          Enable verbose printing\n");
    printf("  -q, --quiet            Suppress all non-error printing\n");
    printf("  -l, --list-matrices    List all available scoring matrices\n");
    printf("  -h, --help             Display this help message\n");
}

void
args_print_config(void)
{
    if (args.quiet)
    {
        return;
    }

    print(SECTION, MSG_NONE, "Configuration");

    print(CONFIG, MSG_LOC(FIRST), "Input: %s", file_name_path(args.path_input));

    if (args.output_file_set)
    {
        if (!args.mode_write)
        {
            print(CONFIG, MSG_LOC(MIDDLE), "Output: Disabled (-W, --no-write), ignoring file");
        }

        else
        {
            print(CONFIG, MSG_LOC(MIDDLE), "Output: %s", file_name_path(args.path_output));
        }
    }

    else
    {
        print(CONFIG, MSG_LOC(MIDDLE), "Output: Disabled (-W, --no-write) or file not specified");
    }

    print(CONFIG, MSG_LOC(MIDDLE), "Sequence type: %s", sequence_type_name(args.seq_type));
    print(CONFIG, MSG_LOC(MIDDLE), "Matrix: %s", matrix_id_name(args.seq_type, args.matrix_id));
    print(CONFIG, MSG_LOC(MIDDLE), "Method: %s", alignment_name(args.method_id));

    if (alignment_linear(args.method_id) && args.gap_penalty_set)
    {
        print(CONFIG, MSG_LOC(MIDDLE), "Gap penalty: %d", args.gap_penalty);
    }

    else if (alignment_affine(args.method_id) && (args.gap_open_set && args.gap_extend_set))
    {
        print(CONFIG, MSG_LOC(MIDDLE), "Gap open: %d, extend: %d", args.gap_open, args.gap_extend);
    }

    if (args.mode_filter)
    {
        print(CONFIG, MSG_LOC(MIDDLE), "Filter threshold: %.1f%%", (double)args.filter * 100.0);
    }

    if (args.mode_write)
    {
        print(CONFIG, MSG_LOC(MIDDLE), "Compression: %u", args.compression_level);
    }

#ifdef USE_CUDA
    if (args.mode_cuda)
    {
        print(CONFIG, MSG_LOC(MIDDLE), "CUDA: Enabled");
    }

    else
    {
        print(CONFIG, MSG_LOC(MIDDLE), "CUDA: Disabled");
    }

#endif

    print(CONFIG, MSG_LOC(LAST), "CPU Threads: %lu", args.thread_num);

    if (args.mode_benchmark)
    {
        print(TIMING, MSG_NONE, "Benchmarking mode enabled");
    }
}

static int
args_parse_scoring_matrix(const char* arg, SequenceType seq_type)
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

static int
args_parse_gap(const char* arg)
{
    char* endptr = NULL;
    long gap = strtol(arg, &endptr, 10);
    if (endptr == arg || *endptr != '\0' || gap < 0 || gap > INT_MAX)
    {
        print(ERROR, MSG_NONE, "Invalid gap value: %s", arg);
        print(INFO, MSG_NONE, "Gap values must be positive integers (auto-negated internally)");
        exit(1);
    }

    return (int)gap;
}

static float
args_parse_filter_threshold(const char* arg)
{
    char* endptr = NULL;
    float threshold = strtof(arg, &endptr);
    if (endptr == arg || *endptr != '\0' || threshold < 0.0f || threshold > 1.0f)
    {
        print(ERROR, MSG_NONE, "Invalid filter threshold: %s", arg);
        print(INFO, MSG_NONE, "Threshold must be between 0.0 and 1.0, representing proportion");
        exit(1);
    }

    return threshold;
}

static unsigned long
args_parse_thread_num(const char* arg)
{
    char* endptr = NULL;
    unsigned long threads = strtoul(arg, &endptr, 10);
    if (endptr == arg || *endptr != '\0')
    {
        print(ERROR, MSG_NONE, "Invalid thread count: %s", arg);
        print(INFO, MSG_NONE, "You can leave out the argument for auto-detection");
        exit(1);
    }

    return threads;
}

static unsigned int
args_parse_compression_level(const char* arg)
{
    char* endptr = NULL;
    unsigned long level = strtoul(arg, &endptr, 10);
    if (endptr == arg || *endptr != '\0' || level > 9)
    {
        print(ERROR, MSG_NONE, "Invalid compression level: %s", arg);
        print(INFO, MSG_NONE, "Compression level must be between 0 (no compression) and 9 (max)");
        exit(1);
    }

    return (unsigned int)level;
}

static void
args_parse(int argc, char* argv[])
{
    int opt;
    int option_index = 0;

    while ((opt = getopt_long(argc, argv, optstring, long_options, &option_index)) != -1)
    {
        switch (opt)
        {
            case 'i':
                if (strlen(optarg) >= MAX_PATH)
                {
                    print(ERROR, MSG_NONE, "Input file path is too long");
                    exit(1);
                }

                snprintf(args.path_input, MAX_PATH, "%s", optarg);
                args.input_file_set = 1;
                break;

            case 't':
                args.seq_type = sequence_type_arg(optarg);
                if (args.seq_type == SEQ_TYPE_INVALID)
                {
                    print(ERROR, MSG_NONE, "Unknown sequence type: %s", optarg);
                    exit(1);
                }

                args.seq_type_set = 1;
                break;

            case 'm':
                if (!args.seq_type_set)
                {
                    print(ERROR, MSG_NONE, "Must specify sequence type (-t) before matrix");
                    exit(1);
                }

                args.matrix_id = args_parse_scoring_matrix(optarg, args.seq_type);
                if (args.matrix_id == PARAM_UNSET)
                {
                    print(ERROR, MSG_NONE, "Unknown scoring matrix: %s", optarg);
                    exit(1);
                }

                args.matrix_set = 1;
                break;

            case 'a':
                args.method_id = alignment_arg(optarg);
                if (args.method_id == ALIGN_INVALID)
                {
                    print(ERROR, MSG_NONE, "Unknown alignment method: %s", optarg);
                    exit(1);
                }

                args.method_id_set = 1;
                break;

            case 'p':
                args.gap_penalty = args_parse_gap(optarg);
                args.gap_penalty_set = 1;
                break;

            case 's':
                args.gap_open = args_parse_gap(optarg);
                args.gap_open_set = 1;
                break;

            case 'e':
                args.gap_extend = args_parse_gap(optarg);
                args.gap_extend_set = 1;
                break;

            case 'o':
                if (strlen(optarg) >= MAX_PATH)
                {
                    print(ERROR, MSG_NONE, "Output file path is too long");
                    exit(1);
                }

                snprintf(args.path_output, MAX_PATH, "%s", optarg);
                args.output_file_set = 1;
                args.mode_write = 1;
                break;

            case 'f':
                args.filter = args_parse_filter_threshold(optarg);
                args.mode_filter = 1;
                break;

            case 'T':
                args.thread_num = args_parse_thread_num(optarg);
                break;

            case 'z':
                args.compression_level = args_parse_compression_level(optarg);
                break;

            case 'B':
                args.mode_benchmark = 1;
                break;

            case 'C':
#ifdef USE_CUDA
                args.mode_cuda = 0;
#else
                print(WARNING, MSG_NONE, "Ignored: -C, --no-cuda (not compiled with CUDA support)");
#endif
                break;

            case 'W':
                args.mode_write = 0;
                break;

            case 'D':
                print_detail_flip();
                break;

            case 'v':
                print_verbose_flip();
                break;

            case 'q':
                args.quiet = 1;
                print_quiet_flip();
                break;

            case 'l':
                args_print_matrices();
                exit(0);

            case 'h':
                args_print_usage(argv[0]);
                exit(0);

            default:
                args_print_usage(argv[0]);
                exit(1);
        }
    }

    if (!args.thread_num)
    {
        args.thread_num = (unsigned long)omp_get_max_threads();
    }

    else if (args.thread_num > INT_MAX)
    {
        print(ERROR, MSG_NONE, "Thread count exceeds maximum of %d", INT_MAX);
        exit(1);
    }

    omp_set_num_threads((int)args.thread_num);

    if (args.method_id == ALIGN_GOTOH_AFFINE && args.gap_open == args.gap_extend)
    {
        if (print_Yn("Equal gap penalties detected, switch to Needleman-Wunsch? (Y/n)"))
        {
            args.method_id = ALIGN_NEEDLEMAN_WUNSCH;
            args.gap_penalty = args.gap_open;
            args.gap_penalty_set = 1;
            args.gap_open = PARAM_UNSET;
            args.gap_extend = PARAM_UNSET;
            args.gap_open_set = 0;
            args.gap_extend_set = 0;
        }
    }
}

void
args_init(int argc, char* argv[])
{
    args.seq_type = SEQ_TYPE_INVALID;
    args.matrix_id = PARAM_UNSET;
    args.method_id = ALIGN_INVALID;

#ifdef USE_CUDA
    args.mode_cuda = 1;
#endif

    print_error_prefix("ARGS");

    args_parse(argc, argv);

    if (!args_validate_required())
    {
        print(SECTION, MSG_NONE, NULL);
        printf("\nPlease check the usage below\n\n");
        args_print_usage(argv[0]);
        exit(1);
    }
}
