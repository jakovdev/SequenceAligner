#include "core/bio/sequence/filtering.h"

#include <stdatomic.h>

#include "core/app/args.h"
#include "core/bio/types.h"
#include "system/arch.h"
#include "system/simd.h"
#include "util/benchmark.h"
#include "util/print.h"
#include "util/progress.h"

static float
similarity_pairwise(sequence_ptr_t seq1, sequence_ptr_t seq2)
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

bool
filter_sequences(sequences_t sequences,
                 sequence_count_t sequence_count,
                 float filter_threshold,
                 bool* keep_flags,
                 sequence_count_t* filtered_count)
{
    if (!sequences || !keep_flags || !filtered_count)
    {
        print(ERROR, MSG_NONE, "Invalid parameters to filter sequences");
        return false;
    }

    if ((sequence_count <= 2) || (filter_threshold <= 0.0f) || (filter_threshold > 1.0f))
    {
        print(ERROR, MSG_NONE, "Invalid sequence count or filter threshold");
        return false;
    }

    const unsigned long num_threads = args_thread_num();
    const size_t expected_progress = sequence_count - 1;

    _Alignas(CACHE_LINE) _Atomic(size_t) global_progress = 0;
    const size_t progress_update_interval = expected_progress / (num_threads * 100);

    for (sequence_count_t i = 0; i < sequence_count; i++)
    {
        keep_flags[i] = true;
    }

    *filtered_count = 0;
    sequence_count_t total_filtered = 0;

    if (!progress_start(&global_progress, expected_progress, "Filtering sequences"))
    {
        return false;
    }

#pragma omp parallel reduction(+ : total_filtered)
    {
        size_t local_progress = 0;
        sequence_count_t local_filtered = 0;

#ifdef _MSC_VER
        int64_t si;
#pragma omp for schedule(dynamic)
        for (si = 1; si < (int64_t)sequence_count; si++)
        {
            sequence_count_t i = (sequence_count_t)si;
#else
#pragma omp for schedule(dynamic)
        for (sequence_count_t i = 1; i < sequence_count; i++)
        {
#endif
            bool should_keep = true;

            for (sequence_count_t j = 0; j < i; j++)
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

            if (++local_progress >= progress_update_interval)
            {
                atomic_fetch_add_explicit(&global_progress, local_progress, memory_order_relaxed);
                local_progress = 0;
            }
        }

        if (local_progress > 0)
        {
            atomic_fetch_add_explicit(&global_progress, local_progress, memory_order_relaxed);
        }

        total_filtered += local_filtered;
    }

    bench_filter_end();

    progress_end();

    bench_filter_start();

    *filtered_count = total_filtered;

    return true;
}
