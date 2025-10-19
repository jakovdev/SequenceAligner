#include "cuda_manager.cuh"
#include "host_interface.h"
#include "host_types.h"

extern "C"
{
    bool cuda_initialize()
    {
        return Cuda::getInstance().initialize();
    }

    bool cuda_triangular(size_t buffer_bytes)
    {
        return !Cuda::getInstance().hasEnoughMemory(buffer_bytes);
    }

    bool cuda_upload_sequences(char* sequences_letters,
                               sequence_offset_t* sequences_offsets,
                               sequence_length_t* sequences_lengths,
                               sequence_count_t sequences_count,
                               size_t total_sequences_length)
    {
        return Cuda::getInstance().uploadSequences(sequences_letters,
                                                   sequences_offsets,
                                                   sequences_lengths,
                                                   sequences_count,
                                                   total_sequences_length);
    }

    bool cuda_upload_scoring(int* scoring_matrix, int* sequence_lookup)
    {
        return Cuda::getInstance().uploadScoring(scoring_matrix, sequence_lookup);
    }

    bool cuda_upload_penalties(int linear, int open, int extend)
    {
        return Cuda::getInstance().uploadPenalties(linear, open, extend);
    }

    bool cuda_upload_triangle_indices(size_t* indices, score_t* score_buffer, size_t buffer_bytes)
    {
        return Cuda::getInstance().uploadTriangleIndices(indices, score_buffer, buffer_bytes);
    }

    bool cuda_kernel_launch(int kernel_id)
    {
        return Cuda::getInstance().launchKernel(kernel_id);
    }

    bool cuda_results_get()
    {
        return Cuda::getInstance().getResults();
    }

    ull cuda_results_progress()
    {
        return Cuda::getInstance().getProgress();
    }

    sll cuda_results_checksum()
    {
        return Cuda::getInstance().getChecksum();
    }

    const char* cuda_error_device_get()
    {
        return Cuda::getInstance().getDeviceError();
    }

    const char* cuda_error_host_get()
    {
        return Cuda::getInstance().getHostError();
    }

    const char* cuda_device_name()
    {
        return Cuda::getInstance().getDeviceName();
    }
}
