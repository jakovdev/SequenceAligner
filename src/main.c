#include "benchmark.h"
#include "csv.h"
#include "h5_handler.h"
#include "print.h"
#include "thread.h"

int
main(int argc, char* argv[])
{
    init_args(argc, argv);

    print(HEADER, MSG_NONE, "SEQUENCE ALIGNER");

    print_config_section();

    print(SECTION, MSG_NONE, "Setting Up Alignment");

    bench_init_start();

    SET_HIGH_CLASS();
    if (!get_mode_multithread())
    {
        PIN_THREAD(0);
    }

    File input_file = get_file(get_input_file_path());
    print(SUCCESS,
          MSG_NONE,
          "Successfully opened input file: %s",
          get_file_name(get_input_file_path()));

    char* current = input_file.file_data;
    char* restrict end = input_file.file_data + input_file.data_size;
    current = parse_header(current, end);

    print(VERBOSE, MSG_LOC(MIDDLE), "Initializing scoring matrix and gap penalties");
    ScoringMatrix scoring = { 0 };
    init_scoring_matrix(&scoring);
    init_gap_penalties();

    print(VERBOSE, MSG_LOC(LAST), "Counting sequences in input file...");
    size_t total_seqs_in_file = count_sequences_in_file(current, end);

    if (total_seqs_in_file == 0)
    {
        print(ERROR, MSG_NONE, "No sequences found in input file");
        free_csv_metadata();
        free_file(&input_file);
        return 0;
    }

    print(DNA, MSG_NONE, "Found %zu sequences", total_seqs_in_file);

    init_seq_pool();

    current = input_file.file_data;
    current = skip_header(current, end);

    print(INFO, MSG_NONE, "Loading sequences from file...");
    SequenceData seqdata = load_sequences_from_file(current, end, total_seqs_in_file);
    if (!seqdata.sequences)
    {
        print(ERROR, MSG_NONE, "Failed to allocate memory for sequences");
        free_csv_metadata();
        free_file(&input_file);
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
    print(CONFIG,
          MSG_LOC(FIRST),
          "Using alignment method: %s",
          get_current_alignment_method_name());

    bench_set_alignments(total_alignments);

    bench_align_start();

    perform_alignments(&h5_handler, seqs, seq_count, total_alignments, &scoring);
    print(SUCCESS, MSG_NONE, "Alignment completed successfully!");

    bench_align_end();

    h5_close(&h5_handler);

    bench_total();

    print(SECTION, MSG_NONE, "Cleaning Up");
    print(VERBOSE, MSG_LOC(FIRST), "Freeing csv metadata");
    free_csv_metadata();
    print(VERBOSE, MSG_LOC(MIDDLE), "Freeing sequence memory pool");
    free_seq_pool();
    print(VERBOSE, MSG_LOC(MIDDLE), "Freeing sequence pointers");
    free(seqs);
    print(VERBOSE, MSG_LOC(LAST), "Closing input file");
    free_file(&input_file);

    print(SUCCESS, MSG_NONE, "All operations completed successfully!");
    print(SECTION, MSG_NONE, NULL);
    return 0;
}