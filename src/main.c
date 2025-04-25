#include "arch.h"
#include "args.h"
#include "benchmark.h"
#include "csv.h"
#include "files.h"
#include "h5_handler.h"
#include "print.h"
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

    seq_pool_init();
    SequenceData seq_data = { 0 };

    {
        CREATE_FILE input_file = { 0 };
        bench_io_add(file_read(&input_file, args_path_input()));

        char* file_cursor = input_file.file_data;
        char* file_end = input_file.file_data + input_file.data_size;

        file_cursor = csv_header_parse(file_cursor, file_end);
        char* file_header_start = file_cursor;
        file_cursor = g_csv_has_no_header ? input_file.file_data : file_header_start;

        print(VERBOSE, MSG_LOC(LAST), "Counting sequences in input file");
        size_t total_seqs_in_file = csv_sequence_lines(file_cursor, file_end);

        if (total_seqs_in_file == 0)
        {
            print(ERROR, MSG_NONE, "No sequences found in input file");
            return 0;
        }

        print(DNA, MSG_NONE, "Found %zu sequences", total_seqs_in_file);

        file_cursor = g_csv_has_no_header ? input_file.file_data : file_header_start;
        bench_io_add(sequences_alloc_from_file(&seq_data, file_cursor, file_end, total_seqs_in_file));

        if (!seq_data.sequences)
        {
            print(ERROR, MSG_NONE, "Failed to allocate memory for sequences");
            return 1;
        }
    }

    size_t seq_count = seq_data.count;
    size_t total_alignments = seq_data.total_alignments;

    H5Handler h5_handler = { 0 };
    bench_io_add(h5_initialize(&h5_handler, seq_count));

    bench_io_add(h5_store_sequences(&h5_handler, seq_data.sequences, seq_count));

    print(SECTION, MSG_NONE, "Performing Alignments");

    print(INFO, MSG_NONE, "Will perform %zu pairwise alignments", total_alignments);

    align(&h5_handler, &seq_data);

    if (!args_mode_write())
    {
        print(INFO, MSG_NONE, "Matrix checksum: %lld", h5_handler.checksum);
    }

    bench_align_end();

    bench_io_add(h5_close(&h5_handler));

    bench_io_end();

    bench_total(total_alignments);

    free(seq_data.sequences);
    return 0;
}