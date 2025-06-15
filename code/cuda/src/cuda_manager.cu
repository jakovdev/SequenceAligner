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
    CUDA_ERROR_CHECK("Failed to get device count");

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
Cuda::uploadSequences(char* seqs, half_t* offsets, half_t* lens, half_t n_seqs, size_t n_chars)
{
    if (!m_initialized || !seqs || !offsets || !lens || !n_seqs || !n_chars)
    {
        setHostError("Invalid parameters for sequence upload");
        return false;
    }

    m_seqs.n_seqs = n_seqs;
    m_seqs.n_letters = n_chars;
    m_results.h_total_count = n_seqs * (n_seqs - 1) / 2;
    constexpr size_t cell_size = sizeof(*m_results.h_scores);

    if (!hasEnoughMemory(n_seqs * n_seqs * cell_size))
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

    m_seqs.constant = canUseConstantMemory();

    if (m_seqs.constant)
    {
        if (!copySequencesToConstantMemory(seqs, offsets, lens))
        {
            return false;
        }
    }

    else
    {
        cudaError_t err;
        DEVICE_MALLOC(m_seqs.d_letters, n_chars, "letter memory");
        DEVICE_MALLOC(m_seqs.d_offsets, n_seqs, "offset memory");
        DEVICE_MALLOC(m_seqs.d_lengths, n_seqs, "length memory");

        HOST_DEVICE_COPY(m_seqs.d_letters, seqs, n_chars, "letters");
        HOST_DEVICE_COPY(m_seqs.d_offsets, offsets, n_seqs, "offsets");
        HOST_DEVICE_COPY(m_seqs.d_lengths, lens, n_seqs, "lengths");
    }

    return true;
}

bool
Cuda::uploadTriangleIndices32(half_t* triangle_indices, int* buffer, size_t buffer_size)
{
    if (!m_initialized || !triangle_indices || !buffer || !buffer_size)
    {
        setHostError("Invalid parameters for storing results, or in --no-write mode");
        return false;
    }

    const size_t expected_size = m_results.h_total_count * sizeof(*buffer);

    if (buffer_size <= expected_size)
    {
        if (buffer_size == expected_size)
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

    m_results.h_scores = buffer;
    m_results.h_indices_32 = triangle_indices;

    if (m_seqs.constant)
    {
        if (!copyTriangleIndicesToConstantMemory(triangle_indices))
        {
            return false;
        }
    }

    else
    {
        cudaError_t err;
        DEVICE_MALLOC(m_seqs.d_indices_32, m_seqs.n_seqs, "offsets");
        HOST_DEVICE_COPY(m_seqs.d_indices_32, triangle_indices, m_seqs.n_seqs, "offsets");
    }

    return true;
}

bool
Cuda::uploadTriangleIndices64(size_t* triangle_indices, int* buffer, size_t buffer_size)
{
    if (!m_initialized || !triangle_indices || !buffer || !buffer_size)
    {
        setHostError("Invalid parameters for storing results, or in --no-write mode");
        return false;
    }

    const size_t expected_size = m_results.h_total_count * sizeof(*buffer);

    if (buffer_size <= expected_size)
    {
        if (buffer_size == expected_size)
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

    m_results.h_scores = buffer;
    m_results.h_indices_64 = triangle_indices;
    m_seqs.indices_64 = true;
    if (!copyTriangleIndices64FlagToConstantMemory(m_seqs.indices_64))
    {
        return false;
    }

    cudaError_t err;
    DEVICE_MALLOC(m_seqs.d_indices_64, m_seqs.n_seqs, "offsets");
    HOST_DEVICE_COPY(m_seqs.d_indices_64, triangle_indices, m_seqs.n_seqs, "offsets");

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
        size_t d_scores_size = 0;

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
            err = cudaDeviceSynchronize();
            CUDA_ERROR_CHECK("Device synchronization failed");
            DEVICE_HOST_COPY(&m_results.h_progress, m_results.d_progress, 1, "progress");

            const size_t n_scores = m_seqs.n_seqs * m_seqs.n_seqs;
            DEVICE_HOST_COPY(m_results.h_scores, m_results.d_scores0, n_scores, "full matrix");
            matrix_copied = true;
        }

        return true;
    }

    if (m_results.h_completed_batch >= m_results.h_total_count)
    {
        return true;
    }

    if (!m_results.h_after_first && m_results.h_batch_size < m_results.h_total_count)
    {
        m_results.h_after_first = true;
        return true;
    }

    size_t batch_offset = m_results.h_completed_batch;
    int* buffer = nullptr;

    if (m_results.h_after_first)
    {
        buffer = (m_results.h_active == 0) ? m_results.d_scores1 : m_results.d_scores0;
    }

    else
    {
        err = cudaDeviceSynchronize();
        CUDA_ERROR_CHECK("Device synchronization failed");
        DEVICE_HOST_COPY(&m_results.h_progress, m_results.d_progress, 1, "progress");
        buffer = m_results.d_scores0;
    }

    size_t n_scores = std::min(m_results.h_batch_size, m_results.h_total_count - batch_offset);

    if (n_scores > 0)
    {
        int* host_scores = m_results.h_scores + batch_offset;
        DEVICE_HOST_COPY(host_scores, buffer, n_scores, "batch scores");
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