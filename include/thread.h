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

// Atomic task counter for work-stealing with block processing
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

INLINE T_Func thread_pool_worker(void* restrict arg) {
    int thread_id = *(int*)arg;
    PIN_THREAD(thread_id);
    
    while (1) {
        sem_wait(g_work_queue.work_ready);
        
        if (!g_work_queue.active) {
            break;
        }
        
        // Process tasks in blocks for better cache locality
        size_t task_block;
        while ((task_block = __atomic_fetch_add(&g_task_counter, TASK_BLOCK_SIZE, __ATOMIC_RELAXED)) 
               < g_work_queue.task_count) {
            
            // Calculate actual block size (handling the last block which might be smaller)
            size_t block_end = task_block + TASK_BLOCK_SIZE;
            if (block_end > g_work_queue.task_count) {
                block_end = g_work_queue.task_count;
            }
            
            // Process block of tasks
            #pragma GCC unroll 8
            for (size_t task_idx = task_block; task_idx < block_end; task_idx++) {
                AlignTask* task = &g_work_queue.tasks[task_idx];
                
                int score = align_sequences(
                    task->seq1, task->len1, 
                    task->seq2, task->len2, 
                    task->scoring
                );
                
                set_matrix_value(g_h5_handler, task->i, task->j, score);
                set_matrix_value(g_h5_handler, task->j, task->i, score);
            }
        }
        
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
    g_threads = (pthread_t*)aligned_alloc(CACHE_LINE, sizeof(pthread_t) * g_num_threads);
    g_thread_ids = (int*)aligned_alloc(CACHE_LINE, sizeof(int) * g_num_threads);
    
    // Create semaphores for work coordination
    g_work_queue.work_ready = (sem_t*)malloc(sizeof(sem_t));
    g_work_queue.work_done = (sem_t*)malloc(sizeof(sem_t));
    sem_init(g_work_queue.work_ready, 0, 0);
    sem_init(g_work_queue.work_done, 0, 0);
    g_work_queue.active = 1;
    
    // Start worker threads
    for (int t = 0; t < g_num_threads; t++) {
        g_thread_ids[t] = t;
        pthread_create(&g_threads[t], NULL, thread_pool_worker, &g_thread_ids[t]);
    }
}

INLINE void submit_tasks(AlignTask* restrict tasks, size_t task_count) {
    g_work_queue.tasks = tasks;
    g_work_queue.task_count = task_count;
    
    // Reset counter for new batch
    __atomic_store_n(&g_task_counter, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&g_threads_waiting, 0, __ATOMIC_RELAXED);
    
    // Signal all threads to start working
    for (int t = 0; t < g_num_threads; t++) {
        sem_post(g_work_queue.work_ready);
    }
    
    // Wait for completion
    sem_wait(g_work_queue.work_done);
}

INLINE void destroy_thread_pool(void) {
    // Signal all threads to exit
    g_work_queue.active = 0;
    for (int t = 0; t < g_num_threads; t++) {
        sem_post(g_work_queue.work_ready);
    }
    
    // Join all threads
    for (int t = 0; t < g_num_threads; t++) {
        pthread_join(g_threads[t], NULL);
    }
    
    // Clean up resources
    sem_destroy(g_work_queue.work_ready);
    sem_destroy(g_work_queue.work_done);
    free(g_work_queue.work_ready);
    free(g_work_queue.work_done);
    aligned_free(g_threads);
    aligned_free(g_thread_ids);
}

#endif