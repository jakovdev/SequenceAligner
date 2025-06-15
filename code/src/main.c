#include "arch.h"
#include "args.h"
#include "benchmark.h"
#include "csv.h"
#include "files.h"
#include "print.h"
#include "seqalign_hdf5.h"
#include "thread.h"

#ifdef USE_CUDA
#include "seqalign_cuda.h"
#endif

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
        CLEANUP(file_text_close) FileText input_file = { 0 };

        bench_io_start();
        file_text_open(&input_file, args_input());
        bench_io_end();

        if (!input_file.text)
        {
            return 1;
        }

        bool headerless;
        int seq_column;

        char* file_cursor = input_file.text;
        char* file_end = input_file.text + input_file.meta.bytes;
        char* file_header_start = csv_header_parse(file_cursor, file_end, &headerless, &seq_column);
        file_cursor = headerless ? input_file.text : file_header_start;

        print(VERBOSE, MSG_LOC(LAST), "Counting sequences in input file");
        sequence_count = csv_total_lines(file_cursor, file_end);

        if (!sequence_count)
        {
            print(ERROR, MSG_NONE, "CSV | No sequences found in input file");
            return 1;
        }

        print(DNA, MSG_NONE, "Found %zu sequences", sequence_count);

        bench_io_start();
        sequences_alloc_from_file(file_cursor, file_end, sequence_count, args_filter(), seq_column);
        bench_io_end();
    }

    if (!sequences_get())
    {
        print(ERROR, MSG_NONE, "SEQUENCES | Failed to allocate memory for sequences");
        return 1;
    }

    sequence_count = sequences_count();
    size_t total_alignments = sequences_alignment_count();

#ifdef USE_CUDA
    cuda_init();
#endif

    bench_io_start();
    if (!h5_open(args_output(), sequence_count, args_compression(), args_mode_write()))
    {
        bench_io_end();
        print(ERROR, MSG_NONE, "HDF5 | Failed to create file, will use no-write mode");

        if (!print_yN("Do you want to continue? [y/N]"))
        {
            print(INFO, MSG_LOC(LAST), "Exiting due to file creation failure");
            h5_close(1);
            return 1;
        }

        bench_io_start();
        h5_open(NULL, 0, 0, false);
    }

    if (!h5_sequences_store(sequences_get(), sequence_count))
    {
        bench_io_end();
        print(ERROR, MSG_NONE, "HDF5 | Failed to store sequences, will use no-write mode");

        if (!print_yN("Do you want to continue? [y/N]"))
        {
            print(INFO, MSG_LOC(LAST), "Exiting due to sequence store failure");
            h5_close(1);
            return 1;
        }

        bench_io_start();
        h5_open(NULL, 0, 0, false);
    }

    bench_io_end();

    print(VERBOSE, MSG_LOC(FIRST), "Initializing substitution matrix");
    scoring_matrix_init();

    print(SECTION, MSG_NONE, "Performing Alignments");

    print(INFO, MSG_NONE, "Will perform %zu pairwise alignments", total_alignments);

    if (!align())
    {
        print(ERROR, MSG_NONE, "Failed to perform alignments");
        h5_close(1);
        return 1;
    }

    if (!args_mode_write())
    {
        print(INFO, MSG_NONE, "Matrix checksum: %lld", h5_checksum());
    }

    bench_print_align();

    bench_io_start();
    h5_close(0);
    bench_io_end();

    bench_print_io();

    bench_print_total(total_alignments);
    return 0;
}