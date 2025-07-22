#pragma once
#ifndef CUDA_MANAGER_HPP
#define CUDA_MANAGER_HPP

#include "host_types.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define R __restrict__
typedef quar_t sequence_length_t;

struct Sequences
{
    Sequences() = default;

    char* d_letters{ nullptr };
    sequence_offset_t* d_offsets{ nullptr };
    sequence_length_t* d_lengths{ nullptr };
    half_t* d_indices_32{ nullptr };
    size_t* d_indices_64{ nullptr };
    sequence_count_t n_seqs{ 0 };
    size_t n_letters{ 0 };
    bool constant{ false };
    bool indices_64{ false };
};

struct KernelResults
{
    KernelResults() = default;

    ~KernelResults()
    {
        cudaFree(d_scores0);
        if (use_batching)
        {
            cudaFree(d_scores1);
        }

        cudaFree(d_progress);
        cudaFree(d_checksum);
    }

    score_t* d_scores0{ nullptr };
    score_t* d_scores1{ nullptr };
    ull* d_progress{ nullptr };
    sll* d_checksum{ nullptr };

    score_t* h_scores{ nullptr };
    half_t* h_indices_32{ nullptr };
    size_t* h_indices_64{ nullptr };
    alignment_size_t h_batch_size{ 0 };
    alignment_size_t h_last_batch{ 0 };
    alignment_size_t h_total_count{ 0 };
    alignment_size_t h_completed_batch{ 0 };
    ull h_progress{ 0 };
    int h_active{ 0 };
    bool use_batching{ false };
    bool h_after_first{ false };
    bool d_triangular{ false };
};

class Cuda
{
  public:
    static Cuda& getInstance();
    Cuda(const Cuda&) = delete;
    Cuda& operator=(const Cuda&) = delete;
    Cuda(Cuda&&) = delete;
    Cuda& operator=(Cuda&&) = delete;

    ~Cuda()
    {
        if (m_initialized)
        {
            if (!m_seqs.constant)
            {
                cudaFree(m_seqs.d_letters);
                cudaFree(m_seqs.d_offsets);
                cudaFree(m_seqs.d_lengths);
                if (m_seqs.indices_64)
                {
                    cudaFree(m_seqs.d_indices_64);
                }

                else
                {
                    cudaFree(m_seqs.d_indices_32);
                }
            }

            cudaDeviceReset();
        }
    }

    bool initialize();
    bool hasEnoughMemory(size_t bytes);

    bool uploadSequences(char* sequences_letters,
                         sequence_offset_t* sequences_offsets,
                         sequence_length_t* sequences_lengths,
                         sequence_count_t sequences_count,
                         size_t total_sequences_length);

    bool uploadScoring(int* scoring_matrix, int* sequence_lookup);
    bool uploadPenalties(int linear, int start, int extend);
    bool uploadTriangleIndices32(half_t* triangle_indices,
                                 score_t* score_buffer,
                                 size_t buffer_bytes);
    bool uploadTriangleIndices64(size_t* triangle_indices,
                                 score_t* score_buffer,
                                 size_t buffer_bytes);

    bool launchKernel(int kernel_id);
    bool getResults();

    ull getProgress();
    sll getChecksum();

    const char* getDeviceError() const;

    const char* getHostError() const { return m_host_error; }

    const char* getDeviceName() const { return m_initialized ? m_device_prop.name : nullptr; }

  private:
    Cuda() = default;

    void setHostError(const char* error) { m_host_error = error; }

    void setDeviceError(cudaError_t error) { m_cuda_error = cudaGetErrorString(error); }

    void setError(const char* host_error, cudaError_t cuda_error)
    {
        setHostError(host_error);
        setDeviceError(cuda_error);
    }

    bool getMemoryStats(size_t* free, size_t* total);
    bool canUseConstantMemory();
    bool copyTriangularMatrixFlag(bool triangular);
    bool copySequencesToConstantMemory(char* seqs,
                                       sequence_offset_t* offsets,
                                       sequence_length_t* lengths);
    bool copyTriangleIndicesToConstantMemory(half_t* indices);
    bool copyTriangleIndices64FlagToConstantMemory(bool indices_64);

    bool switchKernel(int kernel_id);

    cudaDeviceProp m_device_prop;
    KernelResults m_results;
    Sequences m_seqs;
    const char* m_host_error{ "No errors in program" };
    const char* m_cuda_error{ "No errors inside GPU" };
    int m_device_id{ 0 };
    bool m_initialized{ false };
};

#define CUDA_ERROR_CHECK(msg)                                                                      \
    do                                                                                             \
    {                                                                                              \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            setError(msg, err);                                                                    \
            return false;                                                                          \
        }                                                                                          \
    } while (0)

#define DEVICE_MEMSET(ptr, value, size, name)                                                      \
    do                                                                                             \
    {                                                                                              \
        err = cudaMemset(ptr, value, size * sizeof(*ptr));                                         \
        CUDA_ERROR_CHECK("Failed to initialize " name " in GPU");                                  \
    } while (0)

#define DEVICE_MALLOC(ptr, size, name)                                                             \
    do                                                                                             \
    {                                                                                              \
        err = cudaMalloc(&ptr, size * sizeof(*ptr));                                               \
        CUDA_ERROR_CHECK("Failed to allocate " name " to GPU");                                    \
    } while (0)

#define HOST_DEVICE_COPY(dst, src, size, name)                                                     \
    do                                                                                             \
    {                                                                                              \
        static_assert(sizeof(*dst) == sizeof(*src), "Pointer types must match");                   \
        err = cudaMemcpy(dst, src, size * sizeof(*dst), cudaMemcpyHostToDevice);                   \
        CUDA_ERROR_CHECK("Failed to copy " name " to GPU");                                        \
    } while (0)

#define DEVICE_HOST_COPY(dst, src, size, name)                                                     \
    do                                                                                             \
    {                                                                                              \
        static_assert(sizeof(*dst) == sizeof(*src), "Pointer types must match");                   \
        err = cudaMemcpy(dst, src, size * sizeof(*dst), cudaMemcpyDeviceToHost);                   \
        CUDA_ERROR_CHECK("Failed to copy " name " from GPU");                                      \
    } while (0)

#define DEVICE_DEVICE_COPY(dst, src, size, name)                                                   \
    do                                                                                             \
    {                                                                                              \
        static_assert(sizeof(*dst) == sizeof(*src), "Pointer types must match");                   \
        err = cudaMemcpy(dst, src, size * sizeof(*dst), cudaMemcpyDeviceToDevice);                 \
        CUDA_ERROR_CHECK("Failed to copy " name " in GPU");                                        \
    } while (0)

#define CONSTANT_COPY(symbol, src, size, name)                                                     \
    do                                                                                             \
    {                                                                                              \
        err = cudaMemcpyToSymbol(symbol, src, size);                                               \
        CUDA_ERROR_CHECK("Failed to copy " name " to GPU constant memory");                        \
    } while (0)

#endif // CUDA_MANAGER_HPP