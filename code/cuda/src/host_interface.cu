#include "cuda_manager.cuh"
#include "host_interface.h"

extern "C"
{
    bool cuda_initialize()
    {
        return Cuda::getInstance().initialize();
    }

    bool cuda_upload_sequences(char* seqs, half_t* offs, half_t* lens, half_t n_sqs, size_t n_chrs)
    {
        return Cuda::getInstance().uploadSequences(seqs, offs, lens, n_sqs, n_chrs);
    }

    bool cuda_upload_scoring(int* scoring_matrix, int* sequence_lookup)
    {
        return Cuda::getInstance().uploadScoring(scoring_matrix, sequence_lookup);
    }

    bool cuda_upload_penalties(int linear, int open, int extend)
    {
        return Cuda::getInstance().uploadPenalties(linear, open, extend);
    }

    bool cuda_upload_triangle_indices_32(half_t* indices, int* buffer, size_t buffer_size)
    {
        return Cuda::getInstance().uploadTriangleIndices32(indices, buffer, buffer_size);
    }

    bool cuda_upload_triangle_indices_64(size_t* indices, int* buffer, size_t buffer_size)
    {
        return Cuda::getInstance().uploadTriangleIndices64(indices, buffer, buffer_size);
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