#pragma once
#ifndef SEQALIGN_CUDA_H
#define SEQALIGN_CUDA_H

#include "args.h"
#include "benchmark.h"
#include "print.h"
#include "scoring.h"
#include "seqalign_hdf5.h"
#include "sequences.h"

#include "host_interface.h"

#define RETURN_CUDA_ERRORS(error_message)                                                          \
    do                                                                                             \
    {                                                                                              \
        print(ERROR, MSG_LOC(FIRST), error_message);                                               \
        print(ERROR, MSG_LOC(MIDDLE), "CUDA | Host error: %s", cuda_error_host_get());             \
        print(ERROR, MSG_LOC(LAST), "CUDA | Device error: %s", cuda_error_device_get());           \
        return false;                                                                              \
    } while (false)

static inline bool
cuda_init(void)
{
    if (sequences_max_length() > 1024)
    {
        RETURN_CUDA_ERRORS("CUDA | Sequence length exceeds maximum of 1024");
    }

    if (!cuda_initialize())
    {
        RETURN_CUDA_ERRORS("CUDA | Failed to create context");
    }

    const char* device_name = cuda_device_name();
    if (!device_name)
    {
        RETURN_CUDA_ERRORS("CUDA | Failed to query device name");
    }

    print(INFO, MSG_NONE, "Using CUDA device: %s", device_name);
    return true;
}

static inline bool
cuda_align(void)
{
    char* sequences = sequences_flattened();
    half_t* offsets = sequences_offsets();
    half_t* lengths = sequences_lengths();
    half_t sequence_count = (half_t)sequences_count();
    size_t total_length = sequences_total_length();

    if (!cuda_upload_sequences(sequences, offsets, lengths, sequence_count, total_length))
    {
        RETURN_CUDA_ERRORS("CUDA | Failed uploading sequences");
    }

    int FLAT_SCORING_MATRIX[MAX_MATRIX_DIM * MAX_MATRIX_DIM] = { 0 };
    for (size_t i = 0; i < MAX_MATRIX_DIM; i++)
    {
        for (size_t j = 0; j < MAX_MATRIX_DIM; j++)
        {
            FLAT_SCORING_MATRIX[i * MAX_MATRIX_DIM + j] = SCORING_MATRIX[i][j];
        }
    }

    if (!cuda_upload_scoring(FLAT_SCORING_MATRIX, SEQUENCE_LOOKUP))
    {
        RETURN_CUDA_ERRORS("CUDA | Failed uploading scoring data");
    }

    if (!cuda_upload_penalties(args_gap_penalty(), args_gap_open(), args_gap_extend()))
    {
        RETURN_CUDA_ERRORS("CUDA | Failed uploading penalties");
    }

    int* matrix = h5_matrix_data();
    size_t matrix_bytes = h5_matrix_bytes();

    if (h5_triangle_indices_64_bit())
    {
        size_t* result_offsets = h5_triangle_indices_64();
        if (!cuda_upload_triangle_indices_64(result_offsets, matrix, matrix_bytes))
        {
            RETURN_CUDA_ERRORS("CUDA | Failed uploading results storage");
        }
    }

    else
    {
        half_t* result_offsets = h5_triangle_indices_32();
        if (!cuda_upload_triangle_indices_32(result_offsets, matrix, matrix_bytes))
        {
            RETURN_CUDA_ERRORS("CUDA | Failed uploading results storage");
        }
    }

    bench_align_start();

    if (!cuda_kernel_launch(args_align_method()))
    {
        RETURN_CUDA_ERRORS("CUDA | Failed to launch alignment");
    }

    const size_t alignment_count = sequences_alignment_count();
    int percentage = 0;

    print(PROGRESS, MSG_PERCENT(percentage), "Aligning sequences");

    while (true)
    {
        if (!cuda_results_get())
        {
            RETURN_CUDA_ERRORS("CUDA | Failed to get results");
        }

        size_t current_progress = cuda_results_progress();
        percentage = (int)(100 * current_progress / alignment_count);
        print(PROGRESS, MSG_PERCENT(percentage), "Aligning sequences");

        if (current_progress >= alignment_count)
        {
            break;
        }

        else
        {
            if (!cuda_kernel_launch(args_align_method()))
            {
                RETURN_CUDA_ERRORS("CUDA | Failed to launch next batch");
            }
        }
    }

    if (!cuda_results_get())
    {
        RETURN_CUDA_ERRORS("CUDA | Failed to get results");
    }

    if (percentage < 100)
    {
        percentage = 100;
        print(PROGRESS, MSG_PERCENT(percentage), "Aligning sequences");
    }

    h5_checksum_set(cuda_results_checksum() * 2);

    bench_align_end();

    return true;
}

#undef RETURN_CUDA_ERRORS

#endif // SEQALIGN_CUDA_H