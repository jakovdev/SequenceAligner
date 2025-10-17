#include "cuda_manager.cuh"
#include "host_types.h"

constexpr size_t CUDA_BATCH_SIZE = 64ULL << 20;

Cuda&
Cuda::getInstance()
{
    static Cuda instance;
    return instance;
}

bool
Cuda::initialize()
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    CUDA_ERROR_CHECK("Failed to get device count or no CUDA devices available");

    if (device_count == 0)
    {
        setError("No CUDA devices available", err);
        return false;
    }

    err = cudaSetDevice(m_device_id);
    CUDA_ERROR_CHECK("Failed to set CUDA device");

    err = cudaGetDeviceProperties(&m_device_prop, m_device_id);
    CUDA_ERROR_CHECK("Failed to get device properties");

    cudaGetLastError();
    m_initialized = true;

    return true;
}

bool
Cuda::uploadSequences(char* sequences_letters,
                      sequence_offset_t* sequences_offsets,
                      sequence_length_t* sequences_lengths,
                      sequence_count_t sequences_count,
                      size_t total_sequences_length)
{
    if (!m_initialized || !sequences_letters || !sequences_offsets || !sequences_lengths ||
        !sequences_count || !total_sequences_length)
    {
        setHostError("Invalid parameters for sequence upload");
        return false;
    }

    m_seqs.n_seqs = sequences_count;
    m_seqs.n_letters = total_sequences_length;
    m_results.h_total_count = (size_t)sequences_count * (sequences_count - 1) / 2;
    constexpr size_t cell_size = sizeof(*m_results.h_scores);

    if (!hasEnoughMemory(cell_size * sequences_count * sequences_count))
    {
        if (!hasEnoughMemory(m_results.h_total_count * cell_size))
        {
            if (!hasEnoughMemory(CUDA_BATCH_SIZE * cell_size))
            {
                setHostError("Not enough memory for results");
                return false;
            }

            if (!copyTriangularMatrixFlag(true))
            {
                return false;
            }

            m_results.use_batching = true;
            setHostError("No errors in program");
        }

        if (!copyTriangularMatrixFlag(true))
        {
            return false;
        }

        setHostError("No errors in program");
    }

    cudaError_t err;
    DEVICE_MALLOC(m_seqs.d_letters, total_sequences_length, "letter memory");
    DEVICE_MALLOC(m_seqs.d_offsets, sequences_count, "offset memory");
    DEVICE_MALLOC(m_seqs.d_lengths, sequences_count, "length memory");

    HOST_DEVICE_COPY(m_seqs.d_letters, sequences_letters, total_sequences_length, "letters");
    HOST_DEVICE_COPY(m_seqs.d_offsets, sequences_offsets, sequences_count, "offsets");
    HOST_DEVICE_COPY(m_seqs.d_lengths, sequences_lengths, sequences_count, "lengths");

    return true;
}

bool
Cuda::uploadTriangleIndices(size_t* triangle_indices, score_t* score_buffer, size_t buffer_bytes)
{
    if (!m_initialized || !triangle_indices || !score_buffer || !buffer_bytes)
    {
        setHostError("Invalid parameters for storing results, or in --no-write mode");
        return false;
    }

    const size_t expected_size = m_results.h_total_count * sizeof(*score_buffer);

    if (buffer_bytes <= expected_size)
    {
        if (buffer_bytes == expected_size)
        {
            if (!copyTriangularMatrixFlag(true))
            {
                return false;
            }
        }

        else
        {
            setHostError("Buffer size is too small for results");
            return false;
        }
    }

    m_results.h_scores = score_buffer;
    m_results.h_indices = triangle_indices;

    cudaError_t err;
    DEVICE_MALLOC(m_seqs.d_indices, m_seqs.n_seqs, "offsets");
    HOST_DEVICE_COPY(m_seqs.d_indices, triangle_indices, m_seqs.n_seqs, "offsets");

    return true;
}

bool
Cuda::launchKernel(int kernel_id)
{
    if (!m_initialized || !m_seqs.n_seqs || kernel_id > 2 || kernel_id < 0)
    {
        setHostError("Invalid context, sequence data or kernel ID");
        return false;
    }

    if (!m_results.h_after_first)
    {
        cudaError_t err;
        alignment_size_t d_scores_size = 0;

        err = cudaStreamCreate(&m_results.stream0);
        CUDA_ERROR_CHECK("Failed to create compute stream");

        err = cudaStreamCreate(&m_results.stream1);
        CUDA_ERROR_CHECK("Failed to create copy stream");

        if (m_results.d_triangular)
        {
            m_results.h_batch_size = std::min(CUDA_BATCH_SIZE, m_results.h_total_count);
            d_scores_size = m_results.h_batch_size;

            DEVICE_MALLOC(m_results.d_scores0, d_scores_size, "scores");
            DEVICE_MALLOC(m_results.d_scores1, d_scores_size, "scores copy buffer");
        }

        else
        {
            m_results.h_batch_size = m_results.h_total_count;
            d_scores_size = m_seqs.n_seqs * m_seqs.n_seqs;

            DEVICE_MALLOC(m_results.d_scores0, d_scores_size, "scores");
        }

        DEVICE_MALLOC(m_results.d_progress, 1, "progress counter");
        DEVICE_MALLOC(m_results.d_checksum, 1, "checksum");

        DEVICE_MEMSET(m_results.d_progress, 0, 1, "progress counter");
        DEVICE_MEMSET(m_results.d_checksum, 0, 1, "checksum");
    }

    if (!switchKernel(kernel_id))
    {
        return false;
    }

    return true;
}

bool
Cuda::getResults()
{
    if (!m_initialized || !m_results.h_scores)
    {
        setHostError("Invalid context or results storage");
        return false;
    }

    cudaError_t err;

    if (!m_results.d_triangular)
    {
        static bool matrix_copied = false;
        if (!matrix_copied)
        {
            err = cudaStreamSynchronize(m_results.stream0);
            CUDA_ERROR_CHECK("Compute stream synchronization failed");
            DEVICE_HOST_COPY(&m_results.h_progress, m_results.d_progress, 1, "progress");

            const alignment_size_t n_scores = m_seqs.n_seqs * m_seqs.n_seqs;
            DEVICE_HOST_COPY(m_results.h_scores, m_results.d_scores0, n_scores, "full matrix");
            matrix_copied = true;
        }

        return true;
    }

    if (m_results.h_completed_batch >= m_results.h_total_count)
    {
        if (m_results.copy_in_progress)
        {
            err = cudaStreamSynchronize(m_results.stream1);
            CUDA_ERROR_CHECK("Copy stream synchronization failed");
            m_results.copy_in_progress = false;
        }

        return true;
    }

    if (m_results.copy_in_progress)
    {
        err = cudaStreamQuery(m_results.stream1);
        if (err == cudaErrorNotReady)
        {
            return true;
        }

        else if (err != cudaSuccess)
        {
            CUDA_ERROR_CHECK("Copy stream query failed");
        }

        m_results.copy_in_progress = false;
    }

    if (!m_results.h_after_first && m_results.h_batch_size < m_results.h_total_count)
    {
        m_results.h_after_first = true;
        return true;
    }

    alignment_size_t batch_offset = m_results.h_completed_batch;
    score_t* buffer = nullptr;

    if (m_results.h_after_first)
    {
        buffer = (m_results.h_active == 0) ? m_results.d_scores1 : m_results.d_scores0;
    }

    else
    {
        err = cudaStreamSynchronize(m_results.stream0);
        CUDA_ERROR_CHECK("Compute stream synchronization failed");
        DEVICE_HOST_COPY(&m_results.h_progress, m_results.d_progress, 1, "progress");
        buffer = m_results.d_scores0;
    }

    alignment_size_t n_scores = std::min(m_results.h_batch_size,
                                         m_results.h_total_count - batch_offset);

    if (n_scores > 0)
    {
        score_t* host_scores = m_results.h_scores + batch_offset;

        if (m_results.h_after_first)
        {
            DEVICE_HOST_COPY_ASYNC(host_scores,
                                   buffer,
                                   n_scores,
                                   "batch scores",
                                   m_results.stream1);
            m_results.copy_in_progress = true;
        }

        else
        {
            DEVICE_HOST_COPY(host_scores, buffer, n_scores, "batch scores");
        }

        m_results.h_completed_batch += n_scores;
    }

    return true;
}

ull
Cuda::getProgress()
{
    if (!m_initialized)
    {
        return 0;
    }

    return m_results.h_progress;
}

sll
Cuda::getChecksum()
{
    if (!m_initialized || !m_results.d_checksum)
    {
        return 0;
    }

    sll checksum = 0;

    cudaError_t err;

    err = cudaStreamSynchronize(m_results.stream0);
    if (err != cudaSuccess)
    {
        setError("Failed to synchronize compute stream", err);
        return 0;
    }

    DEVICE_HOST_COPY(&checksum, m_results.d_checksum, 1, "checksum");

    return checksum;
}

const char*
Cuda::getDeviceError() const
{
    if (strcmp(m_cuda_error, "no error") == 0)
    {
        return "No errors in GPU";
    }

    return m_cuda_error;
}

bool
Cuda::getMemoryStats(size_t* free, size_t* total)
{
    if (!m_initialized)
    {
        return false;
    }

    cudaError_t err = cudaMemGetInfo(free, total);
    CUDA_ERROR_CHECK("Failed to get memory information");

    return true;
}

bool
Cuda::hasEnoughMemory(size_t bytes)
{
    size_t free = 0;
    size_t total = 0;

    if (!getMemoryStats(&free, &total))
    {
        return false;
    }

    const size_t required_memory = bytes;
    if (free < required_memory * 4 / 3)
    {
        setHostError("Not enough memory for results");
        return false;
    }

    return true;
}