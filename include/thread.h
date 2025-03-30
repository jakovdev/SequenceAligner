#ifndef THREAD_H
#define THREAD_H

#include "seqalign.h"

#define TASK_BLOCK_SIZE (64)

typedef struct {
    const char* seq1;
    const char* seq2;
    size_t len1;
    size_t len2;
    const ScoringMatrix* scoring;
    size_t i;  // Index of first sequence
    size_t j;  // Index of second sequence
} AlignTask;

volatile size_t g_task_counter = 0;
volatile int g_threads_waiting = 0;

typedef struct {
    AlignTask* tasks;
    size_t task_count;
    sem_t* work_ready;
    sem_t* work_done;
    int active;
} WorkQueue;

static WorkQueue g_work_queue;
static pthread_t* g_threads;
static int* g_thread_ids;
static int g_num_threads;
static H5Handler* g_h5_handler;

INLINE T_Func thread_pool_worker(void* arg) {
    int thread_id = *(int*)arg;
    PIN_THREAD(thread_id);
    int64_t local_checksum = 0;
    
    while (1) {
        sem_wait(g_work_queue.work_ready);
        
        if (!g_work_queue.active) break;
        
        // Process tasks in blocks for better cache locality
        size_t task_block;
        while ((task_block = __atomic_fetch_add(&g_task_counter, TASK_BLOCK_SIZE, __ATOMIC_RELAXED)) < g_work_queue.task_count) {
            
            // Calculate actual block size (handling the last block which might be smaller)
            size_t block_end = task_block + TASK_BLOCK_SIZE;
            if (block_end > g_work_queue.task_count) block_end = g_work_queue.task_count;
            
            // Process block of tasks
            #pragma GCC unroll 8
            for (size_t task_idx = task_block; task_idx < block_end; task_idx++) {
                AlignTask* task = &g_work_queue.tasks[task_idx];
                
                int score = align_sequences(
                    task->seq1, task->len1, 
                    task->seq2, task->len2, 
                    task->scoring
                );

                local_checksum += score;
                
                set_matrix_value(g_h5_handler, task->i, task->j, score);
                set_matrix_value(g_h5_handler, task->j, task->i, score);
            }
        }

        __atomic_fetch_add(&g_h5_handler->thread_checksums[thread_id], local_checksum, __ATOMIC_RELAXED);
        local_checksum = 0;
        
        // Signal that this thread is done processing
        if (__atomic_add_fetch(&g_threads_waiting, 1, __ATOMIC_SEQ_CST) == g_num_threads) {
            sem_post(g_work_queue.work_done);
            __atomic_store_n(&g_threads_waiting, 0, __ATOMIC_RELAXED);
        }
    }
    
    T_Ret(NULL);
}

INLINE void init_thread_pool(H5Handler* h5_handler) {
    g_num_threads = get_num_threads();
    g_h5_handler = h5_handler;
    
    // Allocate aligned memory for thread resources
    g_threads = aligned_alloc(CACHE_LINE, sizeof(*g_threads) * g_num_threads);
    g_thread_ids = aligned_alloc(CACHE_LINE, sizeof(*g_thread_ids) * g_num_threads);

    g_h5_handler->thread_checksums = aligned_alloc(CACHE_LINE, sizeof(*g_h5_handler->thread_checksums) * g_num_threads);
    for (int i = 0; i < g_num_threads; i++) g_h5_handler->thread_checksums[i] = 0;
    
    // Create semaphores for work coordination
    g_work_queue.work_ready = malloc(sizeof(*g_work_queue.work_ready));
    g_work_queue.work_done = malloc(sizeof(*g_work_queue.work_done));
    sem_init(g_work_queue.work_ready, 0, 0);
    sem_init(g_work_queue.work_done, 0, 0);
    g_work_queue.active = 1;
    
    // Start worker threads
    for (int t = 0; t < g_num_threads; t++) {
        g_thread_ids[t] = t;
        pthread_create(&g_threads[t], NULL, thread_pool_worker, &g_thread_ids[t]);
    }
}

INLINE void submit_tasks(AlignTask* tasks, size_t task_count) {
    g_work_queue.tasks = tasks;
    g_work_queue.task_count = task_count;
    
    // Reset counters for new batch
    __atomic_store_n(&g_task_counter, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&g_threads_waiting, 0, __ATOMIC_RELAXED);
    
    // Signal all threads to start working
    for (int t = 0; t < g_num_threads; t++) sem_post(g_work_queue.work_ready);
    
    // Wait for completion
    sem_wait(g_work_queue.work_done);
}

INLINE void destroy_thread_pool(void) {
    // Signal all threads to exit
    g_work_queue.active = 0;
    for (int t = 0; t < g_num_threads; t++) sem_post(g_work_queue.work_ready);
    
    // Join all threads
    for (int t = 0; t < g_num_threads; t++) pthread_join(g_threads[t], NULL);
    
    // Clean up resources
    sem_destroy(g_work_queue.work_ready);
    sem_destroy(g_work_queue.work_done);
    free(g_work_queue.work_ready);
    free(g_work_queue.work_done);
    aligned_free(g_threads);
    aligned_free(g_thread_ids);
}

INLINE void perform_alignments(H5Handler* h5_handler, Sequence* seqs, size_t seq_count, size_t total_alignments, const ScoringMatrix* scoring) {
    if (get_mode_multithread()) {
        print_config("Using multithreaded mode with %d threads", get_num_threads());
        init_thread_pool(h5_handler);
        
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
        AlignTask* tasks = huge_page_alloc(tasks_memory_size);
        
        // Process alignments in batches
        size_t processed = 0;
        size_t i = 0, j = 1;  // Start indices for pairwise comparisons
        while (processed < total_alignments) {
            // Determine batch size for this iteration
            size_t batch_size = total_alignments - processed;
            if (batch_size > optimal_batch_size)
                batch_size = optimal_batch_size;
            
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
                tasks[t].scoring = scoring;
                tasks[t].i = i;
                tasks[t].j = j;
                
                // Update indices for next task
                if (++j >= seq_count) {
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
        int64_t local_checksum = 0;
        
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
                    scoring
                );
                
                local_checksum += score;
                
                // Write directly to the in-memory buffer
                set_matrix_value(h5_handler, i, j, score);
                set_matrix_value(h5_handler, j, i, score);
                
                progress_counter++;
                if (progress_counter % progress_step == 0) {
                    print_progress_bar((double)progress_counter / total_alignments, 40, "Aligning sequences");
                }
            }
        }
        h5_handler->matrix_checksum = local_checksum * 2;
        print_progress_bar_end();
    }
}

#endif