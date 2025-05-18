#pragma once
#ifndef THREAD_H
#define THREAD_H

#include "arch.h"
#include "args.h"
#include "benchmark.h"
#include "print.h"
#include "seqalign.h"
#include "seqalign_hdf5.h"

#include <stdatomic.h>

typedef struct
{
    size_t thread_id;
    int64_t local_checksum;
    volatile size_t* current_row;
    pthread_mutex_t* row_mutex;
    atomic_size_t* shared_progress;
    volatile size_t* threads_completed;
    pthread_mutex_t* completion_mutex;
} ThreadStorage;

static inline T_Func
thread_worker(void* thread_arg)
{
    ThreadStorage* storage = CAST(storage)(thread_arg);
    PIN_THREAD(storage->thread_id);

    const size_t sequence_count = sequences_count();
    const size_t alignment_count = sequences_alignment_count();

    size_t local_progress = 0;

    const long thread_num = args_thread_num();
    const size_t num_threads = (thread_num > 0) ? (size_t)thread_num : 1;
    const size_t progress_update_interval = alignment_count / (num_threads * 100);

    const size_t batch_size = num_threads;

    const size_t batch_transition_tail = sequence_count * 9 / 10;

    const size_t batch_size_tail = batch_size / 2;

    while (true)
    {
        size_t current_batch_size;
        size_t row;

        pthread_mutex_lock(storage->row_mutex);
        row = *storage->current_row;

        current_batch_size = (row < batch_transition_tail) ? batch_size : batch_size_tail;

        size_t remaining_rows = sequence_count - row;
        if (remaining_rows < current_batch_size * num_threads / 2)
        {
            current_batch_size = (remaining_rows + num_threads - 1) / num_threads;
            if (current_batch_size == 0)
            {
                current_batch_size = 1;
            }
        }

        *storage->current_row += current_batch_size;
        pthread_mutex_unlock(storage->row_mutex);

        if (row >= sequence_count)
        {
            break;
        }

        size_t end_row = row + current_batch_size;
        if (end_row > sequence_count)
        {
            end_row = sequence_count;
        }

        for (size_t i = row; i < end_row; i++)
        {
            char* seq1;
            size_t len1;

            for (size_t j = i + 1; j < sequence_count; j++)
            {
                char* seq2;
                size_t len2;

                sequences_get_pair(i, &seq1, &len1, j, &seq2, &len2);

                if (UNLIKELY(!seq1 || !seq2 || !len1 || !len2))
                {
                    continue;
                }

                int score = align_pairwise(seq1, (int)len1, seq2, (int)len2);

                storage->local_checksum += score;

                h5_matrix_set(i, j, score);

                local_progress++;

                if (local_progress >= progress_update_interval)
                {
                    atomic_fetch_add(storage->shared_progress, local_progress);
                    local_progress = 0;
                }
            }
        }
    }

    if (local_progress > 0)
    {
        atomic_fetch_add(storage->shared_progress, local_progress);
    }

    if (args_mode_benchmark())
    {
        pthread_mutex_lock(storage->completion_mutex);
        (*storage->threads_completed)++;

        if (*storage->threads_completed == num_threads)
        {
            bench_align_end();
        }

        pthread_mutex_unlock(storage->completion_mutex);
    }

    T_Ret(NULL);
}

static inline bool
align_multithreaded(void)
{
    const long thread_num = args_thread_num();
    const size_t num_threads = (thread_num > 0) ? (size_t)thread_num : 1;

    volatile size_t current_row = 0;
    pthread_mutex_t row_mutex = PTHREAD_MUTEX_INITIALIZER;
    atomic_size_t shared_progress = 0;

    volatile size_t threads_completed = 0;
    pthread_mutex_t completion_mutex = PTHREAD_MUTEX_INITIALIZER;

    pthread_t* threads = MALLOC(threads, num_threads);
    ThreadStorage* thread_storages = MALLOC(thread_storages, num_threads);
    if (!thread_storages || !threads)
    {
        print(ERROR, MSG_NONE, "Failed to allocate memory for multithreading");
        free(thread_storages);
        free(threads);
        pthread_mutex_destroy(&row_mutex);
        pthread_mutex_destroy(&completion_mutex);
        return false;
    }

    bench_align_start();

    for (size_t t = 0; t < num_threads; t++)
    {
        ThreadStorage* storage = &thread_storages[t];
        storage->thread_id = t;
        storage->local_checksum = 0;
        storage->current_row = &current_row;
        storage->row_mutex = &row_mutex;
        storage->shared_progress = &shared_progress;
        storage->threads_completed = &threads_completed;
        storage->completion_mutex = &completion_mutex;

        pthread_create(&threads[t], NULL, thread_worker, storage);
    }

    int percentage = 0;
    print(PROGRESS, MSG_PERCENT(percentage), "Aligning sequences");

    const size_t alignment_count = sequences_alignment_count();
    const unsigned int update_interval_ms = 100;

    for (size_t progress = atomic_load(&shared_progress); progress < alignment_count;
         progress = atomic_load(&shared_progress))
    {
        usleep(update_interval_ms * 1000);
        percentage = (int)(100 * progress / alignment_count);
        print(PROGRESS, MSG_PERCENT(percentage), "Aligning sequences");
    }

    for (size_t t = 0; t < num_threads; t++)
    {
        pthread_join(threads[t], NULL);
    }

    int64_t total_checksum = 0;
    for (size_t t = 0; t < num_threads; t++)
    {
        total_checksum += thread_storages[t].local_checksum;
    }

    h5_checksum_set(total_checksum * 2);

    pthread_mutex_destroy(&row_mutex);
    pthread_mutex_destroy(&completion_mutex);
    free(threads);
    free(thread_storages);

    if (percentage < 100)
    {
        percentage = 100;
        print(PROGRESS, MSG_PERCENT(percentage), "Aligning sequences");
    }

    return true;
}

static inline void
align_singlethreaded(void)
{
    const size_t sequence_count = sequences_count();
    const size_t alignment_count = sequences_alignment_count();
    int64_t local_checksum = 0;
    size_t progress = 0;

    bench_align_start();

    UNROLL(8) for (size_t i = 0; i < sequence_count; i++)
    {
        for (size_t j = i + 1; j < sequence_count; j++)
        {
            char* seq1;
            char* seq2;
            size_t len1, len2;

            sequences_get_pair(i, &seq1, &len1, j, &seq2, &len2);

            int score = align_pairwise(seq1, (int)len1, seq2, (int)len2);

            local_checksum += score;

            h5_matrix_set(i, j, score);

            const int percentage = (int)(100 * ++progress / alignment_count);
            print(PROGRESS, MSG_PERCENT(percentage), "Aligning sequences");
        }
    }

    bench_align_end();

    h5_checksum_set(local_checksum * 2);
}

static inline bool
align(void)
{
    if (args_mode_multithread())
    {
        return align_multithreaded();
    }

    align_singlethreaded();
    return true;
}

#endif // THREAD_H