#include "arch.h"
#include "benchmark.h"
#include "csv.h"
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

    bench_init_start();

    SET_HIGH_CLASS();
    if (!args_mode_multithread())
    {
        PIN_THREAD(0);
    }

    File input_file = file_get(args_path_input());
    print(SUCCESS,
          MSG_NONE,
          "Successfully opened input file: %s",
          file_name_path(args_path_input()));

    char* current = input_file.file_data;
    char* restrict end = input_file.file_data + input_file.data_size;
    current = csv_header_parse(current, end);

    print(VERBOSE, MSG_LOC(MIDDLE), "Initializing scoring matrix and gap penalties");
    ScoringMatrix scoring = { 0 };
    scoring_matrix_init(&scoring);

    print(VERBOSE, MSG_LOC(LAST), "Counting sequences in input file...");
    size_t total_seqs_in_file = csv_sequence_lines(current, end);

    if (total_seqs_in_file == 0)
    {
        print(ERROR, MSG_NONE, "No sequences found in input file");
        csv_metadata_free();
        file_free(&input_file);
        return 0;
    }

    print(DNA, MSG_NONE, "Found %zu sequences", total_seqs_in_file);

    seq_pool_init();

    current = input_file.file_data;
    current = csv_header_skip(current, end);

    print(INFO, MSG_NONE, "Loading sequences from file...");
    SequenceData seqdata = sequences_load_from_file(current, end, total_seqs_in_file);
    if (!seqdata.sequences)
    {
        print(ERROR, MSG_NONE, "Failed to allocate memory for sequences");
        csv_metadata_free();
        file_free(&input_file);
        return 1;
    }

    Sequence* seqs = seqdata.sequences;
    size_t seq_count = seqdata.count;
    size_t total_alignments = seqdata.total_alignments;

    H5Handler h5_handler = h5_initialize(seq_count);

    bench_write_start();

    h5_store_sequences(&h5_handler, seqs, seq_count);

    bench_init_end();

    print(SECTION, MSG_NONE, "Performing Alignments");

    print(INFO, MSG_NONE, "Will perform %zu pairwise alignments", total_alignments);

    bench_set_alignments(total_alignments);

    bench_align_start();

    align(&h5_handler, seqs, seq_count, total_alignments, &scoring);
    print(SUCCESS, MSG_NONE, "Alignment completed successfully!");

    bench_align_end();

    h5_close(&h5_handler);

    bench_total();

    print(SECTION, MSG_NONE, "Cleaning Up");
    print(VERBOSE, MSG_LOC(FIRST), "Freeing csv metadata");
    csv_metadata_free();
    print(VERBOSE, MSG_LOC(MIDDLE), "Freeing sequence memory pool");
    seq_pool_free();
    print(VERBOSE, MSG_LOC(MIDDLE), "Freeing sequence pointers");
    free(seqs);
    print(VERBOSE, MSG_LOC(LAST), "Closing input file");
    file_free(&input_file);

    print(SUCCESS, MSG_NONE, "All operations completed successfully!");
    print(SECTION, MSG_NONE, NULL);
    return 0;
}