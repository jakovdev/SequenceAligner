#include "csv.h"
#include "h5_handler.h"
#include "thread.h"
#include "benchmark.h"
#include "print.h"

int main(int argc, char* argv[]) {
    init_colors();
    print_header("SEQUENCE ALIGNER");
    
    init_args(argc, argv);
    
    print_step_header("Setting Up Alignment");
    
    bench_init_start();

    SET_HIGH_CLASS();
    if (!get_mode_multithread()) PIN_THREAD(0);
    
    File input_file = get_file(get_input_file_path());
    print_success("Successfully opened input file: %s", get_file_name(get_input_file_path()));
    
    char* current = input_file.file_data;
    char* restrict end = input_file.file_data + input_file.data_size;
    current = parse_header(current, end);
    
    print_verbose("Initializing scoring matrix and gap penalties");
    ScoringMatrix scoring;
    init_scoring_matrix(&scoring);
    init_gap_penalties();

    print_verbose("Counting sequences in input file...");
    size_t total_seqs_in_file = count_sequences_in_file(current, end);
    
    if (total_seqs_in_file == 0) {
        print_error("No sequences found in input file");
        free_csv_metadata();
        free_file(&input_file);
        return 0;
    }
    
    print_dna("Found %zu sequences", total_seqs_in_file);

    init_seq_pool();
    
    current = input_file.file_data;
    current = skip_header(current, end);
    
    print_info("Loading sequences from file...");
    SequenceData seqdata = load_sequences_from_file(current, end, total_seqs_in_file);
    if (!seqdata.sequences) {
        print_error("Failed to allocate memory for sequences");
        free_csv_metadata();
        free_file(&input_file);
        return 1;
    }

    Sequence* seqs = seqdata.sequences;
    size_t seq_count = seqdata.count;
    size_t total_alignments = seqdata.total_alignments;
    
    H5Handler h5_handler = init_h5_handler(seq_count);
    
    bench_write_start();
    
    store_sequences_in_h5(&h5_handler, seqs, seq_count);
    
    bench_init_end();
    
    print_step_header("Performing Alignments");
    
    print_info("Will perform %zu pairwise alignments", total_alignments);
    print_config("Using alignment method: %s", get_current_alignment_method_name());

    bench_set_alignments(total_alignments);
    
    bench_align_start();
    
    perform_alignments(&h5_handler, seqs, seq_count, total_alignments, &scoring);
    print_success("Alignment completed successfully!");

    bench_align_end();
    
    close_h5_handler(&h5_handler);
    
    bench_total();
    
    print_step_header("Cleaning Up");
    print_verbose("Freeing csv metadata");
    free_csv_metadata();
    print_verbose("Freeing sequence memory pool");
    free_seq_pool();
    print_verbose("Freeing sequence pointers");
    free(seqs);
    print_verbose("Closing input file");
    free_file(&input_file);
    
    print_success("All operations completed successfully!");
    print_step_header_end(0);
    return 0;
}