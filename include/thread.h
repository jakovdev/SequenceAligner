#ifndef THREAD_H
#define THREAD_H

#include "arch.h"
#include "args.h"
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

    while (1)
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

    T_Ret(NULL);
}

INLINE void
align_multithreaded(H5Handler* h5_handler, SequenceData* seq_data, const ScoringMatrix* scoring)
{
    const int num_threads = args_thread_num();
    const size_t seq_count = seq_data->count;
    const size_t total_alignments = seq_data->total_alignments;

    volatile size_t current_row = 0;
    pthread_mutex_t row_mutex = PTHREAD_MUTEX_INITIALIZER;
    atomic_size_t shared_progress = 0;

    pthread_t* threads = malloc(num_threads * sizeof(*threads));
    ThreadStorage* thread_storages = calloc(num_threads, sizeof(*thread_storages));

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

        pthread_create(&threads[t], NULL, thread_worker, storage);
    }

    double last_update = time_current();
    int progress_percent = 0;

    print(PROGRESS, MSG_PERCENT(progress_percent), "Aligning sequences");

    while (atomic_load(&shared_progress) < total_alignments)
    {
#ifdef _WIN32
        Sleep(10);
#else
        usleep(10000);
#endif

        double now = time_current();
        if (now - last_update > 0.1)
        {
            size_t progress = atomic_load(&shared_progress);
            progress_percent = progress * 100 / total_alignments;

            print(PROGRESS, MSG_PERCENT(progress_percent), "Aligning sequences");

            last_update = now;
        }
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
        align_singlethreaded(h5_handler, seq_data, &scoring);
    }
}

#endif // THREAD_H