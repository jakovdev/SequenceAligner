#include "arch.h"
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
    SequenceData seqdata = { 0 };

    {
        CREATE_FILE input_file = { 0 };
        bench_io_add(file_read(&input_file, args_path_input()));

        char* current = input_file.file_data;
        char* end = input_file.file_data + input_file.data_size;

        current = csv_header_parse(current, end);

        print(VERBOSE, MSG_LOC(LAST), "Counting sequences in input file");
        size_t total_seqs_in_file = csv_sequence_lines(current, end);

        if (total_seqs_in_file == 0)
        {
            print(ERROR, MSG_NONE, "No sequences found in input file");
            return 0;
        }

        print(DNA, MSG_NONE, "Found %zu sequences", total_seqs_in_file);

        current = input_file.file_data;
        current = csv_header_skip(current, end);

        bench_io_add(sequences_alloc_from_file(&seqdata, current, end, total_seqs_in_file));
        if (!seqdata.sequences)
        {
            print(ERROR, MSG_NONE, "Failed to allocate memory for sequences");
            return 1;
        }
    }

    Sequence* seqs = seqdata.sequences;
    size_t seq_count = seqdata.count;
    size_t total_alignments = seqdata.total_alignments;

    H5Handler h5_handler = { 0 };
    bench_io_add(h5_initialize(&h5_handler, seq_count));

    bench_io_add(h5_store_sequences(&h5_handler, seqs, seq_count));

    print(SECTION, MSG_NONE, "Performing Alignments");

    print(INFO, MSG_NONE, "Will perform %zu pairwise alignments", total_alignments);

    bench_align_add(align(&h5_handler, seqs, seq_count, total_alignments));
    print(SUCCESS, MSG_NONE, "Alignment completed successfully!");

    bench_align_end();

    bench_io_add(h5_close(&h5_handler));

    bench_io_end();

    bench_total(total_alignments);

    seq_pool_free();
    free(seqs);

    print(SECTION, MSG_NONE, NULL);
    return 0;
}