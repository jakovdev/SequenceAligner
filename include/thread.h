#ifndef THREAD_H
#define THREAD_H

#include "arch.h"
#include "args.h"
#include "benchmark.h"
#include "h5_handler.h"
#include "print.h"
#include "seqalign.h"
#include <stdatomic.h>

typedef struct
{
    int thread_id;
    H5Handler* h5_handler;
    SequenceData* seq_data;
    const ScoringMatrix* scoring;
    int64_t local_checksum;
    volatile size_t* current_row;
    pthread_mutex_t* row_mutex;
    atomic_size_t* shared_progress;
    volatile int* threads_completed;
    pthread_mutex_t* completion_mutex;
} ThreadStorage;

INLINE T_Func
thread_worker(void* arg)
{
    ThreadStorage* storage = arg;
    PIN_THREAD(storage->thread_id);

    SequenceData* seq_data = storage->seq_data;
    const size_t seq_count = seq_data->count;
    H5Handler* h5_handler = storage->h5_handler;
    const ScoringMatrix* scoring = storage->scoring;

    size_t local_progress = 0;

    const int num_threads = args_thread_num();
    const size_t progress_update_interval = seq_data->total_alignments * 0.01 / num_threads;

    const size_t batch_size = num_threads;

    const size_t batch_transition_tail = seq_count * 9 / 10;

    const size_t batch_size_tail = batch_size / 2;

    while (true)
    {
        size_t current_batch_size;
        size_t row;

        pthread_mutex_lock(storage->row_mutex);
        row = *storage->current_row;

        current_batch_size = (row < batch_transition_tail) ? batch_size : batch_size_tail;

        size_t remaining_rows = seq_count - row;
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

        if (row >= seq_count)
        {
            break;
        }

        size_t end_row = row + current_batch_size;
        if (end_row > seq_count)
        {
            end_row = seq_count;
        }

        for (size_t i = row; i < end_row; i++)
        {
            char* seq1;
            size_t len1;

            for (size_t j = i + 1; j < seq_count; j++)
            {
                char* seq2;
                size_t len2;

                seq_get_pair(seq_data, i, j, &seq1, &len1, &seq2, &len2);

                if (UNLIKELY(!seq1 || !seq2 || !len1 || !len2))
                {
                    continue;
                }

                int score = align_pairwise(seq1, len1, seq2, len2, scoring);

                storage->local_checksum += score;

                h5_set_matrix_value(h5_handler, i, j, score);
                h5_set_matrix_value(h5_handler, j, i, score);

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
            g_times.align = -(g_times.align - time_current());
        }

        pthread_mutex_unlock(storage->completion_mutex);
    }

    T_Ret(NULL);
}

INLINE void
align_multithreaded(H5Handler* h5_handler, SequenceData* seq_data, const ScoringMatrix* scoring)
{
    const int num_threads = args_thread_num();
    const size_t total_alignments = seq_data->total_alignments;

    volatile size_t current_row = 0;
    pthread_mutex_t row_mutex = PTHREAD_MUTEX_INITIALIZER;
    atomic_size_t shared_progress = 0;

    volatile int threads_completed = 0;
    pthread_mutex_t completion_mutex = PTHREAD_MUTEX_INITIALIZER;

    pthread_t* threads = malloc(num_threads * sizeof(*threads));
    ThreadStorage* thread_storages = calloc(num_threads, sizeof(*thread_storages));

    if (args_mode_benchmark())
    {
        g_times.align = time_current();
    }

    for (int t = 0; t < num_threads; t++)
    {
        ThreadStorage* storage = &thread_storages[t];
        storage->thread_id = t;
        storage->h5_handler = h5_handler;
        storage->seq_data = seq_data;
        storage->scoring = scoring;
        storage->local_checksum = 0;
        storage->current_row = &current_row;
        storage->row_mutex = &row_mutex;
        storage->shared_progress = &shared_progress;
        storage->threads_completed = &threads_completed;
        storage->completion_mutex = &completion_mutex;

        pthread_create(&threads[t], NULL, thread_worker, storage);
    }

    int progress_percent = 0;
    print(PROGRESS, MSG_PERCENT(progress_percent), "Aligning sequences");

    for (size_t progress = atomic_load(&shared_progress); progress < total_alignments;
         progress = atomic_load(&shared_progress))
    {
        usleep(100000);
        progress_percent = progress * 100 / total_alignments;
        print(PROGRESS, MSG_PERCENT(progress_percent), "Aligning sequences");
    }

    for (int t = 0; t < num_threads; t++)
    {
        pthread_join(threads[t], NULL);
    }

    int64_t total_checksum = 0;
    for (int t = 0; t < num_threads; t++)
    {
        total_checksum += thread_storages[t].local_checksum;
    }

    h5_handler->checksum = total_checksum * 2;

    pthread_mutex_destroy(&row_mutex);
    pthread_mutex_destroy(&completion_mutex);
    free(threads);
    free(thread_storages);

    if (progress_percent < 100)
    {
        progress_percent = 100;
        print(PROGRESS, MSG_PERCENT(progress_percent), "Aligning sequences");
    }
}

INLINE void
align_singlethreaded(H5Handler* h5_handler, SequenceData* seq_data, const ScoringMatrix* scoring)
{
    size_t seq_count = seq_data->count;
    size_t total_alignments = seq_data->total_alignments;
    size_t progress_counter = 0;
    int64_t local_checksum = 0;

    UNROLL(8) for (size_t i = 0; i < seq_count; i++)
    {
        for (size_t j = i + 1; j < seq_count; j++)
        {
            char* seq1;
            char* seq2;
            size_t len1, len2;

            seq_get_pair(seq_data, i, j, &seq1, &len1, &seq2, &len2);

            int score = align_pairwise(seq1, len1, seq2, len2, scoring);

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
align(H5Handler* h5_handler, SequenceData* seq_data)
{
    print(VERBOSE, MSG_LOC(FIRST), "Initializing scoring matrix");
    ScoringMatrix scoring = { 0 };
    scoring_matrix_init(&scoring);

    if (args_mode_multithread())
    {
        align_multithreaded(h5_handler, seq_data, &scoring);
    }

    else
    {
        double start_time = 0.0;
        if (args_mode_benchmark())
        {
            start_time = time_current();
        }

        align_singlethreaded(h5_handler, seq_data, &scoring);

        if (args_mode_benchmark())
        {
            g_times.align += (time_current() - start_time);
        }
    }
}

#endif // THREAD_H