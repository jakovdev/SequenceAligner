#include "arch.h"
#include "args.h"
#include "benchmark.h"
#include "csv.h"
#include "files.h"
#include "print.h"
#include "seqalign_hdf5.h"
#include "thread.h"

int
main(int argc, char* argv[])
{
    args_init(argc, argv);

    print(HEADER, MSG_NONE, "SEQUENCE ALIGNER");

    args_print_config();

    print(SECTION, MSG_NONE, "Setting Up Alignment");

    SET_HIGH_CLASS();
    if (!args_mode_multithread())
    {
        PIN_THREAD(0);
    }

    size_t sequence_count = 0;

    {
        CLEANUP(file_free) File input_file = { 0 };
        bench_io_add(file_read(&input_file, args_path_input()));

        char* file_cursor = input_file.data;
        char* file_end = input_file.data + input_file.size;
        char* file_header_start = csv_header_parse(file_cursor, file_end);
        file_cursor = g_csv_has_no_header ? input_file.data : file_header_start;

        print(VERBOSE, MSG_LOC(LAST), "Counting sequences in input file");
        sequence_count = csv_total_lines(file_cursor, file_end);

        if (!sequence_count)
        {
            print(ERROR, MSG_NONE, "CSV | No sequences found in input file");
            return 1;
        }

        print(DNA, MSG_NONE, "Found %zu sequences", sequence_count);

        bench_io_add(sequences_alloc_from_file(file_cursor,
                                               file_end,
                                               sequence_count,
                                               args_filter_threshold(),
                                               args_mode_filter(),
                                               g_sequence_column));
    }

    if (!sequences_get())
    {
        print(ERROR, MSG_NONE, "SEQUENCES | Failed to allocate memory for sequences");
        return 1;
    }

    sequence_count = sequences_count();
    size_t total_alignments = sequences_alignment_count();

    bench_io_add(h5_initialize(args_path_output(),
                               sequence_count,
                               args_compression_level(),
                               args_mode_write()));

    bench_io_add(h5_store_sequences(sequences_get(), sequence_count));

    print(VERBOSE, MSG_LOC(FIRST), "Initializing scoring matrix");
    scoring_matrix_init();

    print(SECTION, MSG_NONE, "Performing Alignments");

    print(INFO, MSG_NONE, "Will perform %zu pairwise alignments", total_alignments);

    align();

    if (!args_mode_write())
    {
        print(INFO, MSG_NONE, "Matrix checksum: %lld", h5_get_checksum());
    }

    bench_align_end();

    bench_io_add(h5_close());

    bench_io_end();

    bench_total(total_alignments);
    return 0;
}