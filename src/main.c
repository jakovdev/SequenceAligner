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
    if (!get_mode_multithread()) {
        PIN_THREAD(0);
    }
    
    File file = get_input_file();
    print_success("Successfully opened input file: %s", get_file_name(get_input_file_path()));
    
    char* current = file.file_data;
    char* restrict end = file.file_data + file.data_size;
    current = parse_header(current, end);
    
    print_verbose("Initializing scoring matrix and gap penalties");
    ScoringMatrix scoring;
    init_scoring_matrix(&scoring);
    init_gap_penalties();

    // First pass: count sequences
    size_t total_seqs_in_file = 0;
    size_t seq_count = 0;
    char* count_ptr = current;
    
    print_verbose("Counting sequences in input file...");
    while (count_ptr < end && *count_ptr) {
        if (count_csv_line(&count_ptr)) {
            total_seqs_in_file++;
        }
    }
    
    if (total_seqs_in_file == 0) {
        print_error("No sequences found in input file");
        free_csv_metadata();
        free_input_file(&file);
        return 0;
    }
    
    print_dna("Found %zu sequences", total_seqs_in_file);

    init_seq_pool();
    
    print_verbose("Allocating memory for sequence data");
    Sequence* seqs = (Sequence*)malloc(total_seqs_in_file * sizeof(Sequence));
    if (!seqs) {
        print_error("Failed to allocate memory for sequence pointers");
        free_csv_metadata();
        free_input_file(&file);
        return 1;
    }

    // Second pass: store all sequences
    size_t idx = 0;
    current = file.file_data;
    current = skip_header(current, end);

    size_t filtered_count = 0;
    float filter_threshold = get_filter_threshold();
    bool apply_filtering = get_mode_filter();

    char* temp_seq = NULL;
    size_t temp_seq_capacity = 0;
    
    while (current < end && *current) {
        char* line_end = current;
        while (*line_end && *line_end != '\n' && *line_end != '\r') line_end++;
        size_t max_line_len = line_end - current;
        
        if (max_line_len + 16 > temp_seq_capacity) {
            size_t new_capacity = max_line_len + 64;
            char* new_buffer = (char*)malloc(new_capacity);
            if (!new_buffer) {
                print_error("Failed to allocate memory for sequence");
                free(temp_seq);
                free_csv_metadata();
                free_input_file(&file);
                free_seq_pool();
                free(seqs);
                return 1;
            }
            
            if (temp_seq) {
                free(temp_seq);
            }
            
            temp_seq = new_buffer;
            temp_seq_capacity = new_capacity;
        }
        
        size_t seq_len = parse_csv_line(&current, temp_seq);
        
        if (seq_len == 0) continue;
        
        bool should_include = true;
        
        if (apply_filtering && idx > 0) {
            for (size_t j = 0; j < idx; j++) {
                float similarity = calculate_similarity(temp_seq, seq_len, seqs[j].data, seqs[j].length);
                if (similarity >= filter_threshold) {
                    should_include = false;
                    filtered_count++;
                    break;
                }
            }
        }
        
        if (should_include) {
            init_sequence(&seqs[idx], temp_seq, seq_len);
            idx++;
        }
        
        if ((idx + filtered_count) % 1000 == 0 || current >= end) {
            print_progress_bar((double)(idx + filtered_count) / total_seqs_in_file, 40, apply_filtering ? "Filtering sequences" : "Storing sequences");
        }
    }
    
    free(temp_seq);
    
    seq_count = idx;
    print_progress_bar_end();
    
    if (apply_filtering) {
        print_success("Stored %zu sequences (filtered out %zu)", seq_count, filtered_count);
        
        if (filtered_count > 0 && filtered_count >= total_seqs_in_file / 4) {
            print_verbose("Reallocating memory to save %zu sequence slots", filtered_count);
            Sequence* new_seqs = (Sequence*)realloc(seqs, seq_count * sizeof(Sequence));
            if (new_seqs) {
                seqs = new_seqs;
                print_success("Memory reallocated successfully");
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
    
    bench_write_start();

    store_sequences_in_h5(&h5_handler, seqs, seq_count);
    
    bench_init_end();
    
    print_step_header("Performing Alignments");
    
    print_config("Using alignment method: %s", get_current_alignment_method_name());

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
        print_verbose("Allocating %zu bytes for task memory", tasks_memory_size);
        AlignTask* tasks = (AlignTask*)huge_page_alloc(tasks_memory_size);
        
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
                tasks[t].len1 = seqs[i].length;
                tasks[t].len2 = seqs[j].length;
                tasks[t].scoring = &scoring;
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
    } else {
        print_config("Using single-threaded mode");
        size_t progress_step = total_alignments / 100 + 1;
        size_t progress_counter = 0;
        
        #pragma GCC unroll 8
        for (size_t i = 0; i < seq_count; i++) {
            for (size_t j = i + 1; j < seq_count; j++) {
                // Prefetch next sequences to reduce cache misses
                if (j + 1 < seq_count) {
                    PREFETCH(seqs[i].data + seqs[i].length/2);  // Prefetch middle of current sequence
                    PREFETCH(seqs[j+1].data);                   // Prefetch next sequence
                } else if (i + 1 < seq_count) {
                    PREFETCH(seqs[i+1].data);
                    PREFETCH(seqs[i+2].data);
                }
                
                int score = align_sequences(
                    seqs[i].data, seqs[i].length,
                    seqs[j].data, seqs[j].length,
                    &scoring
                );
                
                // Write directly to the in-memory buffer
                set_matrix_value(&h5_handler, i, j, score);
                set_matrix_value(&h5_handler, j, i, score);
                
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
    free_input_file(&file);
    
    print_success("All operations completed successfully!");
    print_step_header_end(0);
    return 0;
}