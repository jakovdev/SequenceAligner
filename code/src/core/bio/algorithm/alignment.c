#include "core/bio/algorithm/alignment.h"

#include <stdatomic.h>

#include "core/app/args.h"
#include "core/bio/sequence/sequences.h"
#include "core/bio/types.h"
#include "core/interface/seqalign_hdf5.h"
#include "system/arch.h"
#include "util/benchmark.h"
#include "util/progress.h"

bool
align(void)
{
    const sequence_count_t sequence_count = sequences_count();
    const alignment_size_t alignment_count = sequences_alignment_count();
    const unsigned long num_threads = args_thread_num();
    const align_func_t align_func = align_function(args_align_method());

    int64_t global_checksum = 0;
    _Alignas(CACHE_LINE) _Atomic(alignment_size_t) global_progress = 0;
    const alignment_size_t progress_update_interval = MAX(1, alignment_count / (num_threads * 100));

    bench_align_start();

    if (!progress_start(&global_progress, alignment_count, "Aligning sequences"))
    {
        return false;
    }

#pragma omp parallel reduction(+ : global_checksum)
    {
        int64_t local_checksum = 0;
        alignment_size_t local_progress = 0;

#ifdef _MSC_VER
        int64_t si;
#pragma omp for schedule(dynamic)
        for (si = 0; si < (int64_t)sequence_count; si++)
        {
            sequence_count_t i = (sequence_count_t)si;
#else
#pragma omp for schedule(dynamic)
        for (sequence_count_t i = 0; i < sequence_count; i++)
        {
#endif
            for (sequence_count_t j = i + 1; j < sequence_count; j++)
            {
                sequence_ptr_t seq1 = sequence_get(i);
                sequence_ptr_t seq2 = sequence_get(j);

                score_t score = align_func(seq1, seq2);

                local_checksum += score;
                h5_matrix_set(i, j, score);

                local_progress++;
            }

            if (local_progress >= progress_update_interval)
            {
                atomic_fetch_add_explicit(&global_progress, local_progress, memory_order_relaxed);
                local_progress = 0;
            }
        }

        if (local_progress > 0)
        {
            atomic_fetch_add_explicit(&global_progress, local_progress, memory_order_relaxed);
        }

        global_checksum += local_checksum;
    }

    bench_align_end();

    progress_end();

    h5_checksum_set(global_checksum * 2);
    return true;
}
