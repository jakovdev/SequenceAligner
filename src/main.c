#include "csv.h"
#include "h5_handler.h"
#include "thread.h"
#include "benchmark.h"

int main(int argc, char* argv[]) {
    BENCH_INIT_START();
    init_args(argc, argv);
    SET_HIGH_CLASS();
    if (get_mode_multithread()) {
        PIN_THREAD(0);
    }
    
    File file = get_input_file();
    char* current = file.file_data;
    char* restrict end = file.file_data + file.data_size;
    current = skip_header(current, end);
    
    ScoringMatrix scoring;
    init_scoring_matrix(&scoring);

    // First pass: count sequences
    size_t seq_count = 0;
    char* count_ptr = current;
    
    while (count_ptr < end && *count_ptr) {
        char dummy[MAX_SEQ_LEN];
        // TODO: Trim similar sequences for ML
        parse_csv_line(&count_ptr, dummy);
        seq_count++;
    }
    
    if (seq_count == 0) {
        free_input_file(&file);
        printf("No sequences found in input file\n");
        return 0;
    }
    
    printf("Found %zu sequences\n", seq_count);
    size_t total_alignments = (seq_count * (seq_count - 1)) / 2;
    printf("Performing %zu pairwise alignments\n", total_alignments);

    size_t seq_struct_size = ((sizeof(Sequence) + CACHE_LINE - 1) / CACHE_LINE) * CACHE_LINE;
    size_t total_seq_mem = seq_struct_size * seq_count;

    Sequence* seqs = (Sequence*)huge_page_alloc(total_seq_mem);;
    size_t* seq_lens = (size_t*)aligned_alloc(CACHE_LINE, sizeof(size_t) * seq_count);
    
    // Second pass: read all sequences
    size_t idx = 0;
    current = file.file_data;
    current = skip_header(current, end);
    
    while (idx < seq_count && current < end && *current) {
        seq_lens[idx] = parse_csv_line(&current, seqs[idx].data);
        idx++;
    }
    
    H5Handler h5_handler = init_h5_handler(seq_count);
    
    printf("Using %d threads\n", get_num_threads());
    BENCH_INIT_END();
    BENCH_ALIGN_START();
    if (get_mode_multithread()) {
        init_thread_pool(&h5_handler);
        
        // Calculate optimal batch size based on cache size
        size_t optimal_batch_size = L2_CACHE_SIZE / sizeof(AlignTask);
        if (optimal_batch_size > total_alignments) {
            optimal_batch_size = total_alignments;
        }
        
        size_t tasks_memory_size = sizeof(AlignTask) * optimal_batch_size;
        void* task_memory = huge_page_alloc(tasks_memory_size);
        
        AlignTask* tasks = (AlignTask*)task_memory;
        Alignment* results = (Alignment*)aligned_alloc(CACHE_LINE, sizeof(Alignment) * optimal_batch_size);
        
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
            
            submit_tasks(tasks, batch_size, results);
            processed += batch_size;
        }
        
        destroy_thread_pool();
        free(tasks);
        free(results);
    } else {
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
            }
        }
    }
    
    BENCH_ALIGN_END();
    
    int64_t checksum = 0;
    for (size_t i = 0; i < seq_count * seq_count; i++) {
        checksum += h5_handler.buffer.data[i];
    }
    
    close_h5_handler(&h5_handler);
    
    BENCH_TOTAL();

    printf("Matrix checksum: %lld\n", checksum);

    aligned_free(seqs);
    aligned_free(seq_lens);
    free_input_file(&file);
    return 0;
}