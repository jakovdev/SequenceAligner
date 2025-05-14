#include "arch.h"
#include "biotypes.h"
#include "matrices.h"
#include "print.h"

#include <ctype.h>
#include <getopt.h>

#define PARAM_UNSET -1

static struct
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
    float filter;

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
        unsigned mode_multithread : 1;
        unsigned mode_write : 1;
        unsigned mode_benchmark : 1;
        unsigned mode_filter : 1;
        unsigned quiet : 1;
    };
} args = { 0 };

static const char* optstring = "i:o:a:t:m:p:s:e:T:z:f:BWlvqDh";

static struct option long_options[] = { { "input", required_argument, 0, 'i' },
                                        { "output", required_argument, 0, 'o' },
                                        { "align", required_argument, 0, 'a' },
                                        { "type", required_argument, 0, 't' },
                                        { "matrix", required_argument, 0, 'm' },
                                        { "gap-penalty", required_argument, 0, 'p' },
                                        { "gap-start", required_argument, 0, 's' },
                                        { "gap-extend", required_argument, 0, 'e' },
                                        { "threads", required_argument, 0, 'T' },
                                        { "compression", required_argument, 0, 'z' },
                                        { "filter", required_argument, 0, 'f' },
                                        { "benchmark", no_argument, 0, 'B' },
                                        { "no-write", no_argument, 0, 'W' },
                                        { "no-detail", no_argument, 0, 'D' },
                                        { "verbose", no_argument, 0, 'v' },
                                        { "quiet", no_argument, 0, 'q' },
                                        { "help", no_argument, 0, 'h' },
                                        { "list-matrices", no_argument, 0, 'l' },
                                        { 0, 0, 0, 0 } };

#define GETTER(type, name, field)                                                                  \
    type args_##name(void)                                                                         \
    {                                                                                              \
        return field;                                                                              \
    }

GETTER(const char*, input, args.path_input)
GETTER(const char*, output, args.path_output)
GETTER(int, gap_penalty, args.gap_penalty)
GETTER(int, gap_start, args.gap_start)
GETTER(int, gap_extend, args.gap_extend)
GETTER(int, thread_num, args.thread_num)
GETTER(int, align_method, args.method_id)
GETTER(int, sequence_type, args.seq_type)
GETTER(int, scoring_matrix, args.matrix_id)
GETTER(int, compression, args.compression_level)
GETTER(float, filter, args.filter)
GETTER(bool, mode_multithread, args.mode_multithread)
GETTER(bool, mode_benchmark, args.mode_benchmark)
GETTER(bool, mode_filter, args.mode_filter)
GETTER(bool, mode_write, args.mode_write)

#undef GETTER

static int
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

static float
args_parse_filter_threshold(const char* arg)
{
    float threshold = atof(arg);
    if (threshold < 0.0f || threshold > 100.0f)
    {
        return -1.0f;
    }

    return threshold > 1.0f ? threshold / 100.0f : threshold;
}

static int
args_parse_thread_num(const char* arg)
{
    int threads = atoi(arg);
    if (threads < 0)
    {
        return 0;
    }

    return threads;
}

static int
args_parse_compression_level(const char* arg)
{
    int level = atoi(arg);
    return (level < 0 || level > 9) ? 0 : level;
}

static bool
args_validate_file_input(void)
{
    if (!args.input_file_set)
    {
        print(ERROR, MSG_NONE, "ARGS | Missing parameter: input file (-i, --input)");
        return false;
    }

    FILE* test = fopen(args.path_input, "r");
    if (!test)
    {
        print(ERROR, MSG_NONE, "ARGS | Cannot open input file: %s", args.path_input);
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
        print(ERROR, MSG_NONE, "ARGS | Missing parameter: sequence type (-t, --type)");
        valid = false;
    }

    if (!args.method_id_set)
    {
        print(ERROR, MSG_NONE, "ARGS | Missing parameter: alignment method (-a, --align)");
        valid = false;
    }

    if (!args.matrix_set)
    {
        print(ERROR, MSG_NONE, "ARGS | Missing parameter: scoring matrix (-m, --matrix)");
        valid = false;
    }

    if (args.method_id_set && args.method_id >= 0 && args.method_id < ALIGN_COUNT)
    {
        if (alignment_linear(args.method_id) && !args.gap_penalty_set)
        {
            print(ERROR, MSG_NONE, "ARGS | Missing parameter: gap penalty (-p, --gap-penalty)");
            valid = false;
        }

        else if (alignment_affine(args.method_id))
        {
            if (!args.gap_start_set)
            {
                print(ERROR, MSG_NONE, "ARGS | Missing parameter: gap start (-s, --gap-start)");
                valid = false;
            }

            if (!args.gap_extend_set)
            {
                print(ERROR, MSG_NONE, "ARGS | Missing parameter: gap extend (-e, --gap-extend)");
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

    printf("  -a, --align METHOD     Alignment method\n");

    alignment_list();

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

    print(CONFIG, MSG_LOC(MIDDLE), "Method: %s", alignment_name(args.method_id));
    print(CONFIG, MSG_LOC(MIDDLE), "Sequence type: %s", sequence_type_name(args.seq_type));
    print(CONFIG, MSG_LOC(MIDDLE), "Matrix: %s", matrix_id_name(args.seq_type, args.matrix_id));

    if (alignment_linear(args.method_id) && args.gap_penalty_set)
    {
        print(CONFIG, MSG_LOC(MIDDLE), "Gap: %d", args.gap_penalty);
    }

    else if (alignment_affine(args.method_id) && (args.gap_start_set && args.gap_extend_set))
    {
        print(CONFIG, MSG_LOC(MIDDLE), "Gap open: %d, extend: %d", args.gap_start, args.gap_extend);
    }

    if (args.mode_filter)
    {
        print(CONFIG, MSG_LOC(MIDDLE), "Filter threshold: %.1f%%", args.filter * 100.0f);
    }

    if (args.mode_write)
    {
        print(CONFIG, MSG_LOC(MIDDLE), "Compression: %d", args.compression_level);
    }

    print(CONFIG, MSG_LOC(LAST), "Threads: %d", args.thread_num);

    if (args.mode_benchmark)
    {
        print(TIMING, MSG_NONE, "Benchmarking mode enabled");
    }
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
                strncpy(args.path_input, optarg, MAX_PATH - 1);
                args.input_file_set = 1;
                break;

            case 'o':
                strncpy(args.path_output, optarg, MAX_PATH - 1);
                args.output_file_set = 1;
                args.mode_write = 1;
                break;

            case 'a':
                args.method_id = alignment_arg(optarg);
                if (args.method_id != PARAM_UNSET)
                {
                    args.method_id_set = 1;
                }

                else
                {
                    print(ERROR, MSG_NONE, "ARGS | Unknown alignment method: %s", optarg);
                }

                break;

            case 't':
                args.seq_type = sequence_type_arg(optarg);
                if (args.seq_type != PARAM_UNSET)
                {
                    args.seq_type_set = 1;
                }

                else
                {
                    print(ERROR, MSG_NONE, "ARGS | Unknown sequence type: %s", optarg);
                }

                break;

            case 'm':
                if (!args.seq_type_set)
                {
                    print(ERROR, MSG_NONE, "ARGS | Must specify sequence type (-t) before matrix");
                    break;
                }

                args.matrix_id = args_parse_scoring_matrix(optarg, args.seq_type);
                if (args.matrix_id != PARAM_UNSET)
                {
                    args.matrix_set = 1;
                }

                else
                {
                    print(ERROR, MSG_NONE, "ARGS | Unknown scoring matrix: %s", optarg);
                }

                break;

            case 'p':
                args.gap_penalty = atoi(optarg);
                args.gap_penalty_set = 1;
                break;

            case 's':
                args.gap_start = atoi(optarg);
                args.gap_start_set = 1;
                break;

            case 'e':
                args.gap_extend = atoi(optarg);
                args.gap_extend_set = 1;
                break;

            case 'T':
                args.thread_num = args_parse_thread_num(optarg);
                break;

            case 'z':
                args.compression_level = args_parse_compression_level(optarg);
                break;

            case 'f':
                args.filter = args_parse_filter_threshold(optarg);
                if (args.filter >= 0.0f)
                {
                    args.mode_filter = 1;
                }

                else
                {
                    print(ERROR, MSG_NONE, "ARGS | Invalid filter threshold: %s", optarg);
                }

                break;

            case 'B':
                args.mode_benchmark = 1;
                break;

            case 'W':
                args.mode_write = 0;
                break;

            case 'v':
                print_verbose_flip();
                break;

            case 'q':
                args.quiet = 1;
                print_quiet_flip();
                break;

            case 'D':
                print_detail_flip();
                break;

            case 'l':
                args_print_matrices();
                exit(0);

            case 'h':
                args_print_usage(argv[0]);
                exit(0);

            default:
                print(ERROR, MSG_NONE, "ARGS | Unknown option: %c", opt);
                args_print_usage(argv[0]);
                exit(1);
        }
    }

    if (args.thread_num == 0)
    {
        args.thread_num = thread_count();
    }

    args.mode_multithread = (args.thread_num > 1);

    if (args.method_id == ALIGN_GOTOH_AFFINE && args.gap_start == args.gap_extend)
    {
        if (print_Yn("Equal gap penalties detected, switch to Needleman-Wunsch? (Y/n)"))
        {
            args.method_id = ALIGN_NEEDLEMAN_WUNSCH;
            args.gap_penalty = args.gap_start;
            args.gap_penalty_set = 1;
            args.gap_start = PARAM_UNSET;
            args.gap_extend = PARAM_UNSET;
            args.gap_start_set = 0;
            args.gap_extend_set = 0;
        }
    }
}

void
args_init(int argc, char* argv[])
{
    args.method_id = PARAM_UNSET;
    args.seq_type = PARAM_UNSET;
    args.matrix_id = PARAM_UNSET;

    args_parse(argc, argv);

    if (!args_validate_required())
    {
        print(SECTION, MSG_NONE, NULL);
        printf("\nPlease check the usage below\n\n");
        args_print_usage(argv[0]);
        exit(1);
    }
}