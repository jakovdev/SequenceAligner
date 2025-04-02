#ifndef THREAD_H
#define THREAD_H

#include "arch.h"
#include "h5_handler.h"
#include "seqalign.h"

typedef struct
{
    const char* seq1;
    const char* seq2;
    size_t len1;
    size_t len2;
    const ScoringMatrix* scoring;
    size_t i; // First sequence index
    size_t j; // Second sequence index
} AlignTask;

typedef struct
{
    AlignTask* tasks;
    size_t task_count;
    sem_t* work_ready;
    sem_t* work_done;
    int active;
    volatile size_t task_counter;
    volatile int threads_waiting;
} WorkQueue;

typedef struct
{
    pthread_t* threads;
    int* thread_ids;
    int num_threads;
    H5Handler* h5_handler;
    WorkQueue queue;
} ThreadPool;

#define TASK_BLOCK_SIZE (64)

static ThreadPool g_thread_pool;

INLINE T_Func
thread_pool_worker(void* arg)
{
    int thread_id = *(int*)arg;
    PIN_THREAD(thread_id);
    int64_t local_checksum = 0; // This will accumulate across all batches
    WorkQueue* queue = &g_thread_pool.queue;

    while (1)
    {
        sem_wait(queue->work_ready);

        if (!queue->active)
        {
            g_thread_pool.h5_handler->checksums[thread_id] = local_checksum;
            break;
        }

        size_t task_block;
        while ((task_block = __atomic_fetch_add(&queue->task_counter,
                                                TASK_BLOCK_SIZE,
                                                __ATOMIC_RELAXED)) < queue->task_count)
        {

            size_t block_end = min(task_block + TASK_BLOCK_SIZE, queue->task_count);

#pragma GCC unroll 8
            for (size_t task_idx = task_block; task_idx < block_end; task_idx++)
            {
                AlignTask* task = &queue->tasks[task_idx];

                int score = align_pairwise(task->seq1,
                                           task->len1,
                                           task->seq2,
                                           task->len2,
                                           task->scoring);

                local_checksum += score;

                H5Handler* h5_handler = g_thread_pool.h5_handler;
                h5_set_matrix_value(h5_handler, task->i, task->j, score);
                h5_set_matrix_value(h5_handler, task->j, task->i, score);
            }
        }

        if (__atomic_add_fetch(&queue->threads_waiting, 1, __ATOMIC_SEQ_CST) ==
            g_thread_pool.num_threads)
        {
            sem_post(queue->work_done);
            __atomic_store_n(&queue->threads_waiting, 0, __ATOMIC_RELAXED);
        }
    }

    T_Ret(NULL);
}

INLINE void
thread_pool_init(H5Handler* h5_handler)
{
    g_thread_pool.num_threads = args_thread_num();
    g_thread_pool.h5_handler = h5_handler;

    g_thread_pool.threads = aligned_alloc(CACHE_LINE,
                                          sizeof(*g_thread_pool.threads) *
                                              g_thread_pool.num_threads);

    g_thread_pool.thread_ids = aligned_alloc(CACHE_LINE,
                                             sizeof(*g_thread_pool.thread_ids) *
                                                 g_thread_pool.num_threads);

    h5_handler->checksums = aligned_alloc(CACHE_LINE,
                                          sizeof(*h5_handler->checksums) *
                                              g_thread_pool.num_threads);

    for (int i = 0; i < g_thread_pool.num_threads; i++)
    {
        h5_handler->checksums[i] = 0;
    }

    WorkQueue* queue = &g_thread_pool.queue;
    queue->work_ready = malloc(sizeof(*queue->work_ready));
    queue->work_done = malloc(sizeof(*queue->work_done));
    sem_init(queue->work_ready, 0, 0);
    sem_init(queue->work_done, 0, 0);
    queue->active = 1;
    queue->task_counter = 0;
    queue->threads_waiting = 0;

    for (int t = 0; t < g_thread_pool.num_threads; t++)
    {
        g_thread_pool.thread_ids[t] = t;
        pthread_create(&g_thread_pool.threads[t],
                       NULL,
                       thread_pool_worker,
                       &g_thread_pool.thread_ids[t]);
    }
}

INLINE void
tasks_submit(AlignTask* tasks, size_t task_count)
{
    WorkQueue* queue = &g_thread_pool.queue;
    queue->tasks = tasks;
    queue->task_count = task_count;

    __atomic_store_n(&queue->task_counter, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&queue->threads_waiting, 0, __ATOMIC_RELAXED);

    for (int t = 0; t < g_thread_pool.num_threads; t++)
    {
        sem_post(queue->work_ready);
    }

    sem_wait(queue->work_done);
}

INLINE void
thread_pool_free(void)
{
    WorkQueue* queue = &g_thread_pool.queue;

    queue->active = 0;
    for (int t = 0; t < g_thread_pool.num_threads; t++)
    {
        sem_post(queue->work_ready);
    }

    for (int t = 0; t < g_thread_pool.num_threads; t++)
    {
        pthread_join(g_thread_pool.threads[t], NULL);
    }

    sem_destroy(queue->work_ready);
    sem_destroy(queue->work_done);
    free(queue->work_ready);
    free(queue->work_done);
    aligned_free(g_thread_pool.threads);
    aligned_free(g_thread_pool.thread_ids);
}

INLINE void
align_multithreaded(H5Handler* h5_handler,
                    Sequence* seqs,
                    size_t seq_count,
                    size_t total_alignments,
                    const ScoringMatrix* scoring)
{
    thread_pool_init(h5_handler);

    const double ref_small_seqs = 1.0 * KiB;
    const double ref_large_seqs = 32.0 * KiB;
    const double ref_small_bytes = 256 * KiB;
    const double ref_large_bytes = 16.0 * MiB;

    double log_seq_ratio = log10(seq_count / ref_small_seqs) /
                           log10(ref_large_seqs / ref_small_seqs);

    double clamped_ratio = fmax(0.0, fmin(1.0, log_seq_ratio));
    double optimal_memory = ref_small_bytes + clamped_ratio * (ref_large_bytes - ref_small_bytes);

    size_t task_size = sizeof(AlignTask);
    size_t optimal_batch_size = (size_t)(optimal_memory / task_size);
    optimal_batch_size = min(optimal_batch_size, total_alignments);

    print(VERBOSE, MSG_LOC(FIRST), "Batch size: %zu tasks per batch", optimal_batch_size);

    size_t tasks_memory_size = sizeof(AlignTask) * optimal_batch_size;
    AlignTask* tasks = alloc_huge_page(tasks_memory_size);
    size_t processed = 0;
    size_t i = 0, j = 1;

    while (processed < total_alignments)
    {
        size_t batch_size = min(total_alignments - processed, optimal_batch_size);

        for (size_t t = 0; t < batch_size; t++)
        {
#ifdef USE_SIMD
            if (t + PREFETCH_DISTANCE < batch_size)
            {
                size_t prefetch_i = i;
                size_t prefetch_j = j + PREFETCH_DISTANCE;

                if (prefetch_j >= seq_count)
                {
                    prefetch_i++;
                    prefetch_j = prefetch_i + 1;
                }

                if (prefetch_i < seq_count && prefetch_j < seq_count)
                {
                    prefetch(seqs[prefetch_i].data);
                    prefetch(seqs[prefetch_j].data);
                }
            }
#endif

            tasks[t].seq1 = seqs[i].data;
            tasks[t].seq2 = seqs[j].data;
            tasks[t].len1 = seqs[i].length;
            tasks[t].len2 = seqs[j].length;
            tasks[t].scoring = scoring;
            tasks[t].i = i;
            tasks[t].j = j;

            if (++j >= seq_count)
            {
                i++;
                j = i + 1;
            }
        }

        tasks_submit(tasks, batch_size);
        processed += batch_size;
        print(PROGRESS, MSG_PROPORTION((float)processed / total_alignments), "Aligning sequences");
    }

    thread_pool_free();
    aligned_free(tasks);
}

INLINE void
align_signlethreaded(H5Handler* h5_handler,
                     Sequence* seqs,
                     size_t seq_count,
                     const ScoringMatrix* scoring)
{
    size_t total_alignments = seq_count * (seq_count - 1) / 2;
    size_t progress_counter = 0;
    int64_t local_checksum = 0;

#pragma GCC unroll 8
    for (size_t i = 0; i < seq_count; i++)
    {
        for (size_t j = i + 1; j < seq_count; j++)
        {
#ifdef USE_SIMD
            if (j + 1 < seq_count)
            {
                prefetch(seqs[i].data + seqs[i].length / 2);
                prefetch(seqs[j + 1].data);
            }

            else if (i + 1 < seq_count)
            {
                prefetch(seqs[i + 1].data);
                prefetch(seqs[i + 2].data);
            }
#endif

            int score = align_pairwise(seqs[i].data,
                                       seqs[i].length,
                                       seqs[j].data,
                                       seqs[j].length,
                                       scoring);

            local_checksum += score;

            h5_set_matrix_value(h5_handler, i, j, score);
            h5_set_matrix_value(h5_handler, j, i, score);

            progress_counter++;
            print(PROGRESS,
                  MSG_PROPORTION((float)progress_counter / total_alignments),
                  "Aligning sequences");
        }
    }

    h5_handler->checksum = local_checksum * 2;
}

INLINE void
align(H5Handler* h5_handler, Sequence* seqs, size_t seq_count, size_t total_alignments)
{
    print(VERBOSE, MSG_NONE, "Initializing scoring matrix");
    ScoringMatrix scoring = { 0 };
    scoring_matrix_init(&scoring);
    if (args_mode_multithread())
    {
        align_multithreaded(h5_handler, seqs, seq_count, total_alignments, &scoring);
    }

    else
    {
        align_signlethreaded(h5_handler, seqs, seq_count, &scoring);
    }
}

#endif