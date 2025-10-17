#include "cuda_manager.cuh"
#include "host_types.h"

/*
Increase if needed, this depends on your available VRAM.
On an 8GB card, the limit is around 20000-22000.
If over the limit, it will write out of memory error message.
*/
#define MAX_CUDA_SEQUENCE_LENGTH (1024)

/*
Average sequence length, used only for small datasets.
Change if needed, though at best you'll only gain a few microseconds,
since it's only used for small datasets that are already fast to process (miliseconds).
*/
#define LENGTH_AVG (24)

#define KiB (1 << 10)

#define SCORING_MATRIX_DIM (24)
#define SEQUENCE_LOOKUP_SIZE (128)

__constant__ int c_scoring_matrix[SCORING_MATRIX_DIM * SCORING_MATRIX_DIM];
__constant__ int c_sequence_lookup[SEQUENCE_LOOKUP_SIZE];

__constant__ int c_gap_penalty;
__constant__ int c_gap_open;
__constant__ int c_gap_extend;

__constant__ bool c_triangular = false;

#undef KiB

bool
Cuda::copyTriangularMatrixFlag(bool triangular)
{
    if (!m_initialized)
    {
        setHostError("CUDA not initialized");
        return false;
    }

    cudaError_t err;
    m_results.d_triangular = triangular;
    CONSTANT_COPY(c_triangular, &triangular, sizeof(c_triangular), "triangular matrix flag");

    return true;
}

bool
Cuda::uploadScoring(int* scoring_matrix, int* sequence_lookup)
{
    if (!m_initialized)
    {
        setHostError("CUDA not initialized");
        return false;
    }

    cudaError_t err;

    CONSTANT_COPY(c_scoring_matrix, scoring_matrix, sizeof(c_scoring_matrix), "scoring matrix");
    CONSTANT_COPY(c_sequence_lookup, sequence_lookup, sizeof(c_sequence_lookup), "lookup table");

    return true;
}

bool
Cuda::uploadPenalties(int linear, int open, int extend)
{
    if (!m_initialized)
    {
        setHostError("CUDA not initialized");
        return false;
    }

    cudaError_t err;

    CONSTANT_COPY(c_gap_penalty, &linear, sizeof(c_gap_penalty), "gap penalty");
    CONSTANT_COPY(c_gap_open, &open, sizeof(c_gap_open), "gap open penalty");
    CONSTANT_COPY(c_gap_extend, &extend, sizeof(c_gap_extend), "gap extend penalty");

    return true;
}

__forceinline__ __device__ char
d_sequence_char(const Sequences* const seqs, const sequence_index_t ij, const sequence_index_t pos)
{
    return seqs->d_letters[seqs->d_offsets[ij] + pos];
}

__forceinline__ __device__ sequence_length_t
d_sequence_length(const Sequences* const seqs, const sequence_index_t ij)
{
    return seqs->d_lengths[ij];
}

__forceinline__ __device__ size_t
d_triangle_indices(const Sequences* const seqs, const sequence_index_t j)
{
    return seqs->d_indices[j];
}

template<typename T>
__forceinline__ __device__ half_t
d_binary_search(const T* const elements, const half_t length, const size_t target)
{
    half_t low = 1, high = length - 1;
    half_t result = 1;

    while (low <= high)
    {
        half_t mid = (low + high) / 2;

        if (elements[mid] <= target)
        {
            if (mid + 1 >= length || elements[mid + 1] > target)
            {
                result = mid;
                break;
            }

            low = mid + 1;
        }

        else
        {
            high = mid - 1;
        }
    }

    return result;
}

__forceinline__ __device__ sequence_index_t
find_sequence_column_binary_search(const Sequences* const seqs, const alignment_size_t alignment)
{
    return d_binary_search(seqs->d_indices, seqs->n_seqs, alignment);
}

__global__ void
k_nw(const Sequences seqs,
     score_t* R scores,
     ull* R progress,
     sll* R sum,
     alignment_size_t start,
     alignment_size_t batch)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch)
    {
        return;
    }

    const alignment_size_t alignment = start + tid;
    const sequence_count_t n = seqs.n_seqs;

    sequence_index_t i = 0, j = 0;
    j = find_sequence_column_binary_search(&seqs, alignment);
    i = alignment - d_triangle_indices(&seqs, j);

    if (i >= n || j >= n || i >= j)
    {
        // Should never happen
        atomicAdd(progress, 1ULL);
        return;
    }

    const sequence_length_t len1 = d_sequence_length(&seqs, i);
    const sequence_length_t len2 = d_sequence_length(&seqs, j);

    if (len1 > MAX_CUDA_SEQUENCE_LENGTH || len2 > MAX_CUDA_SEQUENCE_LENGTH)
    {
        // Temporary fix, ignore for now
        atomicAdd(progress, 1ULL);
        return;
    }

    score_t score = 0;

    score_t dp_prev[MAX_CUDA_SEQUENCE_LENGTH + 1];
    score_t dp_curr[MAX_CUDA_SEQUENCE_LENGTH + 1];

    for (sequence_length_t col = 0; col <= len2; col++)
    {
        dp_prev[col] = col * -(c_gap_penalty);
    }

    for (sequence_length_t row = 1; row <= len1; ++row)
    {
        dp_curr[0] = row * -(c_gap_penalty);

        for (sequence_length_t col = 1; col <= len2; col++)
        {
            char c1 = d_sequence_char(&seqs, i, row - 1);
            char c2 = d_sequence_char(&seqs, j, col - 1);

            int idx1 = c_sequence_lookup[(unsigned char)c1];
            int idx2 = c_sequence_lookup[(unsigned char)c2];
            score_t match = dp_prev[col - 1] + c_scoring_matrix[idx1 * SCORING_MATRIX_DIM + idx2];

            score_t gap_v = dp_prev[col] - c_gap_penalty;
            score_t gap_h = dp_curr[col - 1] - c_gap_penalty;

            score_t max = match > gap_v ? match : gap_v;
            max = max > gap_h ? max : gap_h;
            dp_curr[col] = max;
        }

        for (half_t col = 0; col <= len2; col++)
        {
            dp_prev[col] = dp_curr[col];
        }
    }

    score = dp_prev[len2];

    if (!c_triangular)
    {
        scores[(size_t)i * n + j] = score;
        scores[(size_t)j * n + i] = score;
    }

    else
    {
        scores[tid] = score;
    }

    atomicAdd(progress, 1ULL);
    atomicAdd(reinterpret_cast<ull*>(sum), static_cast<ull>(score));
}

__global__ void
k_ga(const Sequences seqs,
     score_t* R scores,
     ull* R progress,
     sll* R sum,
     alignment_size_t start,
     alignment_size_t batch)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch)
    {
        return;
    }

    const alignment_size_t alignment = start + tid;
    const sequence_count_t n = seqs.n_seqs;

    sequence_index_t i = 0, j = 0;
    j = find_sequence_column_binary_search(&seqs, alignment);
    i = alignment - d_triangle_indices(&seqs, j);

    if (i >= n || j >= n || i >= j)
    {
        // Should never happen
        atomicAdd(progress, 1ULL);
        return;
    }

    const sequence_length_t len1 = d_sequence_length(&seqs, i);
    const sequence_length_t len2 = d_sequence_length(&seqs, j);

    if (len1 > MAX_CUDA_SEQUENCE_LENGTH || len2 > MAX_CUDA_SEQUENCE_LENGTH)
    {
        // Temporary fix, ignore for now
        atomicAdd(progress, 1ULL);
        return;
    }

    score_t score = 0;

    score_t match[MAX_CUDA_SEQUENCE_LENGTH + 1];
    score_t gap_x[MAX_CUDA_SEQUENCE_LENGTH + 1];
    score_t gap_y[MAX_CUDA_SEQUENCE_LENGTH + 1];

    match[0] = 0;
    gap_x[0] = gap_y[0] = SCORE_MIN;

    for (sequence_length_t col = 1; col <= len2; col++)
    {
        gap_x[col] = max(match[col - 1] - c_gap_open, gap_x[col - 1] - c_gap_extend);
        match[col] = gap_x[col];
        gap_y[col] = SCORE_MIN;
    }

    score_t prev_match[MAX_CUDA_SEQUENCE_LENGTH + 1];
    score_t prev_gap_y[MAX_CUDA_SEQUENCE_LENGTH + 1];

    for (sequence_length_t col = 0; col <= len2; col++)
    {
        prev_match[col] = match[col];
        prev_gap_y[col] = gap_y[col];
    }

    for (sequence_length_t row = 1; row <= len1; ++row)
    {
        match[0] = row * -(c_gap_penalty);
        gap_x[0] = SCORE_MIN;
        gap_y[0] = max(prev_match[0] - c_gap_open, prev_gap_y[0] - c_gap_extend);
        match[0] = gap_y[0];

        char c1 = d_sequence_char(&seqs, i, row - 1);
        int idx1 = c_sequence_lookup[(unsigned char)c1];

        for (sequence_length_t col = 1; col <= len2; col++)
        {
            char c2 = d_sequence_char(&seqs, j, col - 1);
            int idx2 = c_sequence_lookup[(unsigned char)c2];
            score_t similarity = c_scoring_matrix[idx1 * SCORING_MATRIX_DIM + idx2];

            score_t diag_score = prev_match[col - 1] + similarity;

            score_t open_x = match[col - 1] - c_gap_open;
            score_t extend_x = gap_x[col - 1] - c_gap_extend;
            gap_x[col] = max(open_x, extend_x);

            score_t open_y = prev_match[col] - c_gap_open;
            score_t extend_y = prev_gap_y[col] - c_gap_extend;
            gap_y[col] = max(open_y, extend_y);

            match[col] = max(diag_score, max(gap_x[col], gap_y[col]));
        }

        for (sequence_length_t col = 0; col <= len2; col++)
        {
            prev_match[col] = match[col];
            prev_gap_y[col] = gap_y[col];
        }
    }

    score = match[len2];

    if (!c_triangular)
    {
        scores[(size_t)i * n + j] = score;
        scores[(size_t)j * n + i] = score;
    }

    else
    {
        scores[tid] = score;
    }

    atomicAdd(progress, 1ULL);
    atomicAdd(reinterpret_cast<ull*>(sum), static_cast<ull>(score));
}

__global__ void
k_sw(const Sequences seqs,
     score_t* R scores,
     ull* R progress,
     sll* R sum,
     alignment_size_t start,
     alignment_size_t batch)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch)
    {
        return;
    }

    const alignment_size_t alignment = start + tid;
    const sequence_count_t n = seqs.n_seqs;

    sequence_index_t i = 0, j = 0;
    j = find_sequence_column_binary_search(&seqs, alignment);
    i = alignment - d_triangle_indices(&seqs, j);

    if (i >= n || j >= n || i >= j)
    {
        // Should never happen
        atomicAdd(progress, 1ULL);
        return;
    }

    const sequence_length_t len1 = d_sequence_length(&seqs, i);
    const sequence_length_t len2 = d_sequence_length(&seqs, j);

    if (len1 > MAX_CUDA_SEQUENCE_LENGTH || len2 > MAX_CUDA_SEQUENCE_LENGTH)
    {
        // Temporary fix, ignore for now
        atomicAdd(progress, 1ULL);
        return;
    }

    score_t score = 0;

    score_t match[MAX_CUDA_SEQUENCE_LENGTH + 1];
    score_t gap_x[MAX_CUDA_SEQUENCE_LENGTH + 1];
    score_t gap_y[MAX_CUDA_SEQUENCE_LENGTH + 1];

    for (sequence_length_t col = 0; col <= len2; col++)
    {
        match[col] = 0;
        gap_x[col] = gap_y[col] = SCORE_MIN;
    }

    score_t prev_match[MAX_CUDA_SEQUENCE_LENGTH + 1];
    score_t prev_gap_y[MAX_CUDA_SEQUENCE_LENGTH + 1];

    for (sequence_length_t col = 0; col <= len2; col++)
    {
        prev_match[col] = match[col];
        prev_gap_y[col] = gap_y[col];
    }

    score_t max_score = 0;

    for (sequence_length_t row = 1; row <= len1; ++row)
    {
        match[0] = 0;
        gap_x[0] = gap_y[0] = SCORE_MIN;

        char c1 = d_sequence_char(&seqs, i, row - 1);
        int idx1 = c_sequence_lookup[(unsigned char)c1];

        for (half_t col = 1; col <= len2; col++)
        {
            char c2 = d_sequence_char(&seqs, j, col - 1);
            int idx2 = c_sequence_lookup[(unsigned char)c2];
            score_t similarity = c_scoring_matrix[idx1 * SCORING_MATRIX_DIM + idx2];

            score_t diag_score = prev_match[col - 1] + similarity;

            score_t open_x = match[col - 1] - c_gap_open;
            score_t extend_x = gap_x[col - 1] - c_gap_extend;
            gap_x[col] = max(open_x, extend_x);

            score_t open_y = prev_match[col] - c_gap_open;
            score_t extend_y = prev_gap_y[col] - c_gap_extend;
            gap_y[col] = max(open_y, extend_y);

            score_t best = max(0, max(diag_score, max(gap_x[col], gap_y[col])));
            match[col] = best;

            if (best > max_score)
            {
                max_score = best;
            }
        }

        for (sequence_length_t col = 0; col <= len2; col++)
        {
            prev_match[col] = match[col];
            prev_gap_y[col] = gap_y[col];
        }
    }

    score = max_score;

    if (!c_triangular)
    {
        scores[(size_t)i * n + j] = score;
        scores[(size_t)j * n + i] = score;
    }

    else
    {
        scores[tid] = score;
    }

    atomicAdd(progress, 1ULL);
    atomicAdd(reinterpret_cast<ull*>(sum), static_cast<ull>(score));
}

bool
Cuda::switchKernel(int kernel_id)
{
    cudaError_t err;
    alignment_size_t offset = m_results.h_last_batch;
    alignment_size_t batch = m_results.h_batch_size;
    score_t* d_buffer = m_results.d_scores0;

    if (offset >= m_results.h_total_count)
    {
        if (m_results.h_after_first)
        {
            err = cudaDeviceSynchronize();
            CUDA_ERROR_CHECK("Device synchronization failed on final check");
            DEVICE_HOST_COPY(&m_results.h_progress, m_results.d_progress, 1, "final progress");
            m_results.h_active = 1 - m_results.h_active;
        }

        return true;
    }

    if (m_results.d_triangular)
    {
        offset = m_results.h_last_batch;
        if (offset + batch > m_results.h_total_count)
        {
            batch = m_results.h_total_count - offset;
        }

        if (!batch)
        {
            if (m_results.h_after_first)
            {
                err = cudaDeviceSynchronize();
                CUDA_ERROR_CHECK("Device synchronization failed on final check");
                DEVICE_HOST_COPY(&m_results.h_progress, m_results.d_progress, 1, "final progress");
            }

            return true;
        }

        if (m_results.h_after_first)
        {
            err = cudaDeviceSynchronize();
            CUDA_ERROR_CHECK("Device synchronization failed");
            DEVICE_HOST_COPY(&m_results.h_progress, m_results.d_progress, 1, "progress");
            m_results.h_active = 1 - m_results.h_active;
        }

        if (m_results.h_active == 1)
        {
            d_buffer = m_results.d_scores1;
        }
    }

    const int blockDim = m_device_prop.maxThreadsPerBlock;
    const int gridDim = (int)((batch + blockDim - 1) / blockDim);

    if (gridDim <= 0 || gridDim > m_device_prop.maxGridSize[0])
    {
        setHostError("Grid size exceeds device limit");
        return false;
    }

    ull* d_progress = m_results.d_progress;
    sll* d_checksum = m_results.d_checksum;

    switch (kernel_id)
    {
        default:
            setHostError("Invalid kernel ID");
            return false;
        case 0: // Gotoh Affine
            k_ga<<<gridDim, blockDim, 0, m_results.stream0>>>(m_seqs,
                                                              d_buffer,
                                                              d_progress,
                                                              d_checksum,
                                                              offset,
                                                              batch);
            break;

        case 1: // Needleman-Wunsch
            k_nw<<<gridDim, blockDim, 0, m_results.stream0>>>(m_seqs,
                                                              d_buffer,
                                                              d_progress,
                                                              d_checksum,
                                                              offset,
                                                              batch);
            break;

        case 2: // Smith-Waterman
            k_sw<<<gridDim, blockDim, 0, m_results.stream0>>>(m_seqs,
                                                              d_buffer,
                                                              d_progress,
                                                              d_checksum,
                                                              offset,
                                                              batch);
            break;
    }

    err = cudaGetLastError();
    CUDA_ERROR_CHECK("Kernel launch failed");

    m_results.h_last_batch += batch;

    return true;
}
