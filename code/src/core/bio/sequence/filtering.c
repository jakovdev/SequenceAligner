#include "core/bio/sequence/filtering.h"

#include <stdatomic.h>

#include "core/app/args.h"
#include "core/bio/types.h"
#include "system/arch.h"
#include "util/print.h"

typedef struct
{
    unsigned long thread_id;
    sequences_t sequences;
    sequence_count_t sequence_count;
    float filter_threshold;
    volatile sequence_index_t* current_index;
    pthread_mutex_t* index_mutex;
    bool* keep_flags;
    atomic_size_t* shared_progress;
    sequence_count_t* filtered_count;
    pthread_mutex_t* filtered_count_mutex;
} FilterThreadStorage;

static float
similarity_pairwise(const sequence_ptr_t seq1, const sequence_ptr_t seq2)
{
    if (UNLIKELY(!seq1->length || !seq2->length))
    {
        return 0.0f;
    }

    sequence_length_t min_len = seq1->length < seq2->length ? seq1->length : seq2->length;
    size_t matches = 0;

#ifdef USE_SIMD
    sequence_length_t vec_limit = (min_len / BYTES) * BYTES;

    for (sequence_length_t i = 0; i < vec_limit; i += BYTES * 2)
    {
        prefetch(seq1->letters + i + BYTES);
        prefetch(seq2->letters + i + BYTES);
    }

    for (sequence_length_t i = 0; i < vec_limit; i += BYTES)
    {
        veci_t v1 = loadu((const veci_t*)(seq1->letters + i));
        veci_t v2 = loadu((const veci_t*)(seq2->letters + i));

#if defined(__AVX512F__) && defined(__AVX512BW__)
        num_t mask = cmpeq_epi8(v1, v2);
        matches += (size_t)__builtin_popcountll(mask);
#else
        num_t mask = movemask_epi8(cmpeq_epi8(v1, v2));
        matches += (size_t)__builtin_popcount(mask);
#endif
    }

    for (sequence_length_t i = vec_limit; i < min_len; i++)
    {
        matches += (seq1->letters[i] == seq2->letters[i]);
    }

#else
    for (sequence_length_t i = 0; i < min_len; i++)
    {
        matches += (seq1->letters[i] == seq2->letters[i]);
    }

#endif

    return (float)matches / (float)min_len;
}

static inline T_Func
filter_thread_worker(void* thread_arg)
{
    FilterThreadStorage* storage = CAST(storage)(thread_arg);
    // PIN_THREAD(storage->thread_id);

    const sequence_count_t sequence_count = storage->sequence_count;
    const float filter_threshold = storage->filter_threshold;
    sequences_t sequences = storage->sequences;
    bool* keep_flags = storage->keep_flags;

    const unsigned long thread_num = args_thread_num();
    const unsigned long num_threads = (thread_num > 0) ? thread_num : 1;
    const sequence_count_t batch_size = (sequence_count_t)num_threads;
    const sequence_count_t progress_update_interval = sequence_count / (batch_size * 100);

    sequence_count_t local_progress = 0;
    sequence_count_t local_filtered = 0;

    while (true)
    {
        sequence_index_t start_index;
        sequence_count_t current_batch_size = batch_size;

        pthread_mutex_lock(storage->index_mutex);
        start_index = *storage->current_index;

        sequence_count_t remaining = sequence_count - start_index;
        if (remaining < current_batch_size * batch_size / 2)
        {
            current_batch_size = (remaining + batch_size - 1) / batch_size;
            if (current_batch_size == 0)
            {
                current_batch_size = 1;
            }
        }

        *storage->current_index += current_batch_size;
        pthread_mutex_unlock(storage->index_mutex);

        if (start_index >= sequence_count)
        {
            break;
        }

        sequence_count_t end_index = start_index + current_batch_size;
        if (end_index > sequence_count)
        {
            end_index = sequence_count;
        }

        for (sequence_index_t i = start_index; i < end_index; i++)
        {
            if (!keep_flags[i])
            {
                local_progress++;
                continue;
            }

            bool should_keep = true;

            for (sequence_index_t j = 0; j < i; j++)
            {
                if (!keep_flags[j])
                {
                    continue;
                }

                float similarity = similarity_pairwise(&sequences[i], &sequences[j]);
                if (similarity >= filter_threshold)
                {
                    should_keep = false;
                    local_filtered++;
                    break;
                }
            }

            keep_flags[i] = should_keep;
            local_progress++;

            if (local_progress >= progress_update_interval)
            {
                atomic_fetch_add(storage->shared_progress, local_progress);
                local_progress = 0;
            }
        }
    }

    if (local_progress > 0)
    {
        atomic_fetch_add(storage->shared_progress, local_progress);
    }

    pthread_mutex_lock(storage->filtered_count_mutex);
    (*storage->filtered_count) += local_filtered;
    pthread_mutex_unlock(storage->filtered_count_mutex);

    T_Ret(NULL);
}

bool
filter_sequences_multithreaded(sequences_t sequences,
                               sequence_count_t sequence_count,
                               float filter_threshold,
                               bool* keep_flags,
                               sequence_count_t* filtered_count)
{
    const unsigned long thread_num = args_thread_num();
    const unsigned long num_threads = (thread_num > 0) ? thread_num : 1;

    volatile sequence_index_t current_index = 1; // keep first sequence
    pthread_mutex_t index_mutex = PTHREAD_MUTEX_INITIALIZER;
    atomic_size_t shared_progress = 0;

    *filtered_count = 0;
    pthread_mutex_t filtered_count_mutex = PTHREAD_MUTEX_INITIALIZER;

    pthread_t* threads = MALLOC(threads, num_threads);
    FilterThreadStorage* thread_storages = MALLOC(thread_storages, num_threads);

    if (!thread_storages || !threads)
    {
        print(ERROR, MSG_NONE, "Failed to allocate memory for threads");
        free(thread_storages);
        free(threads);
        pthread_mutex_destroy(&index_mutex);
        pthread_mutex_destroy(&filtered_count_mutex);
        return false;
    }

    for (sequence_count_t i = 0; i < sequence_count; i++)
    {
        keep_flags[i] = true;
    }

    for (unsigned long t = 0; t < num_threads; t++)
    {
        FilterThreadStorage* storage = &thread_storages[t];
        storage->thread_id = t;
        storage->sequences = sequences;
        storage->sequence_count = sequence_count;
        storage->filter_threshold = filter_threshold;
        storage->current_index = &current_index;
        storage->index_mutex = &index_mutex;
        storage->keep_flags = keep_flags;
        storage->shared_progress = &shared_progress;
        storage->filtered_count = filtered_count;
        storage->filtered_count_mutex = &filtered_count_mutex;

        pthread_create(&threads[t], NULL, filter_thread_worker, storage);
        if (!threads[t])
        {
            print(ERROR, MSG_NONE, "Failed to create thread %lu", t);

            for (unsigned long j = 0; j < t; j++)
            {
                if (threads[j])
                {
                    pthread_join(threads[j], NULL);
                }
            }

            free(threads);
            free(thread_storages);
            pthread_mutex_destroy(&index_mutex);
            pthread_mutex_destroy(&filtered_count_mutex);
            return false;
        }
    }

    int percentage = 0;
    print(PROGRESS, MSG_PERCENT(percentage), "Filtering sequences");

    const unsigned int update_interval_ms = 100;
    const sequence_count_t expected_progress = sequence_count > 1 ? sequence_count - 1 : 0;

    for (size_t progress = atomic_load(&shared_progress); progress < expected_progress;
         progress = atomic_load(&shared_progress))
    {
        usleep(update_interval_ms * 1000);
        percentage = (int)(100 * progress / expected_progress);
        print(PROGRESS, MSG_PERCENT(percentage), "Filtering sequences");
    }

    for (unsigned long t = 0; t < num_threads; t++)
    {
        pthread_join(threads[t], NULL);
    }

    pthread_mutex_destroy(&index_mutex);
    pthread_mutex_destroy(&filtered_count_mutex);
    free(threads);
    free(thread_storages);

    if (percentage < 100)
    {
        percentage = 100;
        print(PROGRESS, MSG_PERCENT(percentage), "Filtering sequences");
    }

    return true;
}

void
filter_sequences_singlethreaded(sequences_t sequences,
                                sequence_count_t sequence_count,
                                float filter_threshold,
                                bool* keep_flags,
                                sequence_count_t* filtered_count)
{
    print(PROGRESS, MSG_PERCENT(0), "Filtering sequences");

    for (sequence_count_t i = 0; i < sequence_count; i++)
    {
        keep_flags[i] = true;
    }

    *filtered_count = 0;

    for (sequence_count_t i = 1; i < sequence_count; i++)
    {
        for (sequence_count_t j = 0; j < i; j++)
        {
            if (!keep_flags[j])
            {
                continue;
            }

            float similarity = similarity_pairwise(&sequences[i], &sequences[j]);
            if (similarity >= filter_threshold)
            {
                keep_flags[i] = false;
                (*filtered_count)++;
                break;
            }
        }

        int percentage = (int)(100 * i / sequence_count);
        print(PROGRESS, MSG_PERCENT(percentage), "Filtering sequences");
    }
}