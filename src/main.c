#include "csv.h"
#include "h5_handler.h"
#include "thread.h"
#include "benchmark.h"
#include "print.h"

int main(int argc, char* argv[]) {
    bench_init_start();

    init_colors();
    print_header("SEQUENCE ALIGNER");
    
    init_args(argc, argv);
    
    print_step_header("Setting Up Alignment");
    
    SET_HIGH_CLASS();
    if (!get_mode_multithread()) {
        PIN_THREAD(0);
    }
    
    File file = get_input_file();
    print_success("Successfully opened input file: %s", get_file_name(get_input_file_path()));
    
    char* current = file.file_data;
    char* restrict end = file.file_data + file.data_size;
    current = parse_header(current, end);
    
    ScoringMatrix scoring;
    init_scoring_matrix(&scoring);
    print_verbose("Initialized scoring matrix");

    // First pass: count sequences
    size_t total_seqs_in_file = 0;
    size_t seq_count = 0;
    char* count_ptr = current;
    
    print_verbose("Counting sequences in input file...");
    while (count_ptr < end && *count_ptr) {
        char dummy[MAX_SEQ_LEN];
        parse_csv_line(&count_ptr, dummy);
        total_seqs_in_file++;
    }
    
    if (total_seqs_in_file == 0) {
        print_error("No sequences found in input file");
        free_csv_metadata();
        free_input_file(&file);
        return 0;
    }
    
    print_dna("Found %zu sequences", total_seqs_in_file);

    size_t seq_struct_size = ((sizeof(Sequence) + CACHE_LINE - 1) / CACHE_LINE) * CACHE_LINE;
    size_t total_seq_mem = seq_struct_size * total_seqs_in_file;
    print_verbose("Allocating %zu bytes for sequence data", total_seq_mem);

    Sequence* seqs = (Sequence*)huge_page_alloc(total_seq_mem);
    size_t* seq_lens = (size_t*)aligned_alloc(CACHE_LINE, sizeof(size_t) * total_seqs_in_file);
    
    // Second pass: store all sequences
    size_t idx = 0;
    current = file.file_data;
    current = skip_header(current, end);

    size_t filtered_count = 0;
    float filter_threshold = get_filter_threshold();
    bool apply_filtering = get_mode_filter();

    while (current < end && *current) {
        char temp_seq[MAX_SEQ_LEN];
        size_t seq_len = parse_csv_line(&current, temp_seq);
        
        if (seq_len == 0) continue;
        
        bool should_include = true;
        
        if (apply_filtering && idx > 0) {
            for (size_t j = 0; j < idx; j++) {
                float similarity = calculate_similarity(temp_seq, seq_len, seqs[j].data, seq_lens[j]);
                if (similarity >= filter_threshold) {
                    should_include = false;
                    filtered_count++;
                    break;
                }
            }
        }
        
        if (should_include) {
            memcpy(seqs[idx].data, temp_seq, seq_len);
            seqs[idx].data[seq_len] = '\0';
            seq_lens[idx] = seq_len;
            idx++;
        }
        
        if ((idx + filtered_count) % 1000 == 0 || current >= end) {
            print_progress_bar((double)(idx + filtered_count) / total_seqs_in_file, 40, apply_filtering ? "Filtering sequences" : "Storing sequences");
        }
    }
    
    seq_count = idx;
    print_progress_bar_end();
    
    if (apply_filtering) {
        print_success("Stored %zu sequences (filtered out %zu)", seq_count, filtered_count);
        
        if (filtered_count > 0 && filtered_count >= total_seqs_in_file / 4) {
            print_verbose("Reallocating memory to save %zu sequence slots (%.2f MiB)", filtered_count, (filtered_count * seq_struct_size) / (1024.0 * 1024.0));
            size_t new_seq_mem = seq_struct_size * seq_count;
            Sequence* new_seqs = (Sequence*)huge_page_alloc(new_seq_mem);
            if (new_seqs) {
                memcpy(new_seqs, seqs, new_seq_mem);
                aligned_free(seqs);
                seqs = new_seqs;
                size_t* new_seq_lens = (size_t*)aligned_alloc(CACHE_LINE, sizeof(size_t) * seq_count);
                if (new_seq_lens) {
                    memcpy(new_seq_lens, seq_lens, sizeof(size_t) * seq_count);
                    aligned_free(seq_lens);
                    seq_lens = new_seq_lens;
                    print_success("Memory reallocated, now using %zu bytes for sequences", new_seq_mem);
                }
            }
        } else {
            print_verbose("Using %zu of %zu sequence slots after filtering (%.2f%% eff)", seq_count, total_seqs_in_file, (seq_count * 100.0) / total_seqs_in_file);
        }
    } else {
        print_success("Stored %zu sequences", seq_count);
    }

    size_t total_alignments = (seq_count * (seq_count - 1)) / 2;
    print_info("Will perform %zu pairwise alignments", total_alignments);
    bench_set_alignments(total_alignments);
    
    H5Handler h5_handler = init_h5_handler(seq_count);
    
    print_config("Using alignment method: %s", get_current_alignment_method_name());
    
    bench_init_end();
    
    print_step_header("Performing Alignments");
    
    bench_align_start();
    
    if (get_mode_multithread()) {
        print_config("Using multithreaded mode with %d threads", get_num_threads());
        init_thread_pool(&h5_handler);
        
        // Calculate optimal batch size based on testing data
        size_t task_size = sizeof(AlignTask);
        // Reference points for interpolation:
        // - For 1024 sequences: use L2_CACHE_SIZE (256KiB)
        // - For 32768 sequences: use 16MiB of tasks
        const double ref_small_seqs = 1.0 * KiB;
        const double ref_large_seqs = 32.0 * KiB;
        const double ref_small_bytes = L2_CACHE_SIZE;
        const double ref_large_bytes = 16.0 * MiB;
        
        double log_seq_ratio = log10(seq_count / ref_small_seqs) / log10(ref_large_seqs / ref_small_seqs);
        double clamped_ratio = log_seq_ratio < 0.0 ? 0.0 : (log_seq_ratio > 1.0 ? 1.0 : log_seq_ratio);
        double optimal_memory = ref_small_bytes + clamped_ratio * (ref_large_bytes - ref_small_bytes);
        size_t optimal_batch_size = (size_t)(optimal_memory / task_size);
        optimal_batch_size = optimal_batch_size < total_alignments ? optimal_batch_size : total_alignments;
        print_verbose("Batch size: %zu tasks per batch", optimal_batch_size);
        
        size_t tasks_memory_size = sizeof(AlignTask) * optimal_batch_size;
        void* task_memory = huge_page_alloc(tasks_memory_size);
        print_verbose("Allocated %zu bytes for task memory", tasks_memory_size);
        
        AlignTask* tasks = (AlignTask*)task_memory;
        Alignment* results = (Alignment*)aligned_alloc(CACHE_LINE, sizeof(Alignment) * optimal_batch_size);
        print_verbose("Allocated %zu bytes for result memory", sizeof(Alignment) * optimal_batch_size);
        
        // Process alignments in batches
        size_t processed = 0;
        size_t i = 0, j = 1;  // Start indices for pairwise comparisons
        while (processed < total_alignments) {
            // Determine batch size for this iteration
            size_t batch_size = total_alignments - processed;
            if (batch_size > optimal_batch_size) {
                batch_size = optimal_batch_size;
            }
            
            // Fill the task queue
            for (size_t t = 0; t < batch_size; t++) {
                // Prefetch sequences for next task to reduce cache misses
                // TODO: Benchmark
                if (t + PREFETCH_DISTANCE < batch_size) {
                    size_t prefetch_i = i;
                    size_t prefetch_j = j + PREFETCH_DISTANCE;
                    
                    // Calculate next indices
                    if (prefetch_j >= seq_count) {
                        prefetch_i++;
                        prefetch_j = prefetch_i + 1;
                    }
                    
                    if (prefetch_i < seq_count && prefetch_j < seq_count) {
                        PREFETCH(seqs[prefetch_i].data);
                        PREFETCH(seqs[prefetch_j].data);
                    }
                }
                
                // Set up the current task
                tasks[t].seq1 = seqs[i].data;
                tasks[t].seq2 = seqs[j].data;
                tasks[t].len1 = seq_lens[i];
                tasks[t].len2 = seq_lens[j];
                tasks[t].scoring = &scoring;
                tasks[t].result = &results[t];
                tasks[t].i = i;
                tasks[t].j = j;
                
                // Update indices for next task
                j++;
                if (j >= seq_count) {
                    i++;
                    j = i + 1;
                }
            }
            
            submit_tasks(tasks, batch_size);
            processed += batch_size;
            print_progress_bar((double)processed / total_alignments, 40, "Aligning sequences");
        }
        print_progress_bar_end();
        print_verbose("Destroying thread pool");
        destroy_thread_pool();
        print_verbose("Freeing task memory");
        aligned_free(tasks);
        print_verbose("Freeing result memory");
        aligned_free(results);
    } else {
        print_config("Using single-threaded mode");
        size_t progress_step = total_alignments / 100 + 1;
        size_t progress_counter = 0;
        
        #pragma GCC unroll 8
        for (size_t i = 0; i < seq_count; i++) {
            for (size_t j = i + 1; j < seq_count; j++) {
                // Prefetch next sequences to reduce cache misses
                if (j + 1 < seq_count) {
                    PREFETCH(seqs[i].data + seq_lens[i]/2);  // Prefetch middle of current sequence
                    PREFETCH(seqs[j+1].data);                // Prefetch next sequence
                } else if (i + 1 < seq_count) {
                    PREFETCH(seqs[i+1].data);
                    PREFETCH(seqs[i+2].data);
                }
                
                Alignment result = align_sequences(
                    seqs[i].data, seq_lens[i],
                    seqs[j].data, seq_lens[j],
                    &scoring
                );
                
                // Write directly to the in-memory buffer
                set_matrix_value(&h5_handler, i, j, result.score);
                set_matrix_value(&h5_handler, j, i, result.score);
                
                progress_counter++;
                if (progress_counter % progress_step == 0) {
                    print_progress_bar((double)progress_counter / total_alignments, 40, "Aligning sequences");
                }
            }
        }
        print_progress_bar_end();
    }
    
    print_success("Alignment completed successfully!");
    bench_align_end();
    
    print_step_header("Finalizing Results");
    
    int64_t checksum = 0;
    for (size_t i = 0; i < seq_count * seq_count; i++) {
        checksum += h5_handler.buffer.data[i];
    }

    print_info("Matrix checksum: %lld", checksum);
    
    close_h5_handler(&h5_handler);
    
    bench_total();
    
    print_step_header("Cleaning Up");
    print_verbose("Freeing csv metadata");
    free_csv_metadata();
    print_verbose("Freeing sequence data");
    aligned_free(seqs);
    print_verbose("Freeing sequence lengths");
    aligned_free(seq_lens);
    print_verbose("Closing input file");
    free_input_file(&file);
    
    print_success("All operations completed successfully!");
    print_step_header_end();
    return 0;
}