#include "cuda_manager.cuh"
#include "host_types.h"

constexpr size_t CUDA_BATCH_SIZE = 64ULL << 20;

Cuda &Cuda::getInstance()
{
	static Cuda instance;
	return instance;
}

bool Cuda::initialize()
{
	int device_count = 0;
	cudaError_t err = cudaGetDeviceCount(&device_count);
	CUDA_ERROR("Failed to get device count or none available");

	if (device_count == 0) {
		setError("No CUDA devices available", err);
		return false;
	}

	err = cudaSetDevice(m_id);
	CUDA_ERROR("Failed to set CUDA device");

	err = cudaGetDeviceProperties(&m_dev, m_id);
	CUDA_ERROR("Failed to get device properties");

	cudaGetLastError();
	m_init = true;

	return true;
}

bool Cuda::uploadSequences(char *sequences_letters,
			   sequence_offset_t *sequences_offsets,
			   sequence_length_t *sequences_lengths,
			   sequence_count_t sequences_count,
			   size_t total_sequences_length)
{
	if (!m_init || !sequences_letters || !sequences_offsets ||
	    !sequences_lengths || !sequences_count || !total_sequences_length) {
		setHostError("Invalid parameters for sequence upload");
		return false;
	}

	m_seqs.n_seqs = sequences_count;
	m_seqs.n_letters = total_sequences_length;
	m_kr.h_total_count =
		(size_t)sequences_count * (sequences_count - 1) / 2;
	constexpr size_t cell_size = sizeof(*m_kr.h_scores);

	if (!hasEnoughMemory(cell_size * sequences_count * sequences_count)) {
		if (!hasEnoughMemory(m_kr.h_total_count * cell_size)) {
			if (!hasEnoughMemory(CUDA_BATCH_SIZE * cell_size)) {
				setHostError("Not enough memory for results");
				return false;
			}

			if (!copyTriangularMatrixFlag(true))
				return false;

			m_kr.use_batching = true;
			setHostError("No errors in program");
		}

		if (!copyTriangularMatrixFlag(true))
			return false;

		setHostError("No errors in program");
	}

	cudaError_t err;
	D_MALLOC(m_seqs.d_letters, total_sequences_length);
	D_MALLOC(m_seqs.d_offsets, sequences_count);
	D_MALLOC(m_seqs.d_lengths, sequences_count);

	HD_COPY(m_seqs.d_letters, sequences_letters, total_sequences_length);
	HD_COPY(m_seqs.d_offsets, sequences_offsets, sequences_count);
	HD_COPY(m_seqs.d_lengths, sequences_lengths, sequences_count);

	return true;
}

bool Cuda::uploadTriangleIndices(size_t *triangle_indices,
				 score_t *score_buffer, size_t buffer_bytes)
{
	if (!m_init || !triangle_indices || !score_buffer || !buffer_bytes) {
		setHostError(
			"Invalid parameters for uploading indices, or using --no-write");
		return false;
	}

	const size_t expected_size = m_kr.h_total_count * sizeof(*score_buffer);

	if (buffer_bytes <= expected_size) {
		if (buffer_bytes == expected_size) {
			if (!copyTriangularMatrixFlag(true))
				return false;
		} else {
			setHostError("Buffer size is too small for results");
			return false;
		}
	}

	m_kr.h_scores = score_buffer;
	m_kr.h_indices = triangle_indices;

	cudaError_t err;
	D_MALLOC(m_seqs.d_indices, m_seqs.n_seqs);
	HD_COPY(m_seqs.d_indices, triangle_indices, m_seqs.n_seqs);

	return true;
}

bool Cuda::launchKernel(int kernel_id)
{
	if (!m_init || !m_seqs.n_seqs || kernel_id > 2 || kernel_id < 0) {
		setHostError("Invalid context, sequence data or kernel ID");
		return false;
	}

	if (!m_kr.h_after_first) {
		cudaError_t err;
		alignment_size_t d_scores_size = 0;

		err = cudaStreamCreate(&m_kr.s_comp);
		CUDA_ERROR("Failed to create compute stream");

		err = cudaStreamCreate(&m_kr.s_copy);
		CUDA_ERROR("Failed to create copy stream");

		if (m_kr.d_triangular) {
			m_kr.h_batch_size =
				std::min(CUDA_BATCH_SIZE, m_kr.h_total_count);
			d_scores_size = m_kr.h_batch_size;

			D_MALLOC(m_kr.d_scores0, d_scores_size);
			D_MALLOC(m_kr.d_scores1, d_scores_size);
		} else {
			m_kr.h_batch_size = m_kr.h_total_count;
			d_scores_size = m_seqs.n_seqs * m_seqs.n_seqs;

			D_MALLOC(m_kr.d_scores0, d_scores_size);
		}

		D_MALLOC(m_kr.d_progress, 1);
		D_MALLOC(m_kr.d_checksum, 1);

		D_MEMSET(m_kr.d_progress, 0, 1);
		D_MEMSET(m_kr.d_checksum, 0, 1);
	}

	if (!switchKernel(kernel_id))
		return false;

	return true;
}

bool Cuda::getResults()
{
	if (!m_init || !m_kr.h_scores) {
		setHostError("Invalid context or results storage");
		return false;
	}

	cudaError_t err;

	if (!m_kr.d_triangular) {
		static bool matrix_copied = false;
		if (!matrix_copied) {
			err = cudaStreamSynchronize(m_kr.s_comp);
			CUDA_ERROR("Compute stream synchronization failed");
			DH_COPY(&m_kr.h_progress, m_kr.d_progress, 1);

			const alignment_size_t n_scores =
				m_seqs.n_seqs * m_seqs.n_seqs;
			DH_COPY(m_kr.h_scores, m_kr.d_scores0, n_scores);
			matrix_copied = true;
		}

		return true;
	}

	if (m_kr.h_completed_batch >= m_kr.h_total_count) {
		if (m_kr.copy_in_progress) {
			err = cudaStreamSynchronize(m_kr.s_copy);
			CUDA_ERROR("Copy stream synchronization failed");
			m_kr.copy_in_progress = false;
		}

		return true;
	}

	if (m_kr.copy_in_progress) {
		err = cudaStreamQuery(m_kr.s_copy);
		if (err == cudaErrorNotReady)
			return true;
		else if (err != cudaSuccess)
			CUDA_ERROR("Copy stream query failed");

		m_kr.copy_in_progress = false;
	}

	if (!m_kr.h_after_first && m_kr.h_batch_size < m_kr.h_total_count) {
		m_kr.h_after_first = true;
		return true;
	}

	alignment_size_t batch_offset = m_kr.h_completed_batch;
	score_t *buffer = nullptr;

	if (m_kr.h_after_first) {
		buffer = (m_kr.h_active == 0) ? m_kr.d_scores1 : m_kr.d_scores0;
	} else {
		err = cudaStreamSynchronize(m_kr.s_comp);
		CUDA_ERROR("Compute stream synchronization failed");
		DH_COPY(&m_kr.h_progress, m_kr.d_progress, 1);
		buffer = m_kr.d_scores0;
	}

	alignment_size_t n_scores =
		std::min(m_kr.h_batch_size, m_kr.h_total_count - batch_offset);

	if (n_scores > 0) {
		score_t *scores = m_kr.h_scores + batch_offset;

		if (m_kr.h_after_first) {
			DH_COPY_ASYNC(scores, buffer, n_scores, m_kr.s_copy);
			m_kr.copy_in_progress = true;
		} else {
			DH_COPY(scores, buffer, n_scores);
		}

		m_kr.h_completed_batch += n_scores;
	}

	return true;
}

ull Cuda::getProgress()
{
	if (!m_init)
		return 0;

	return m_kr.h_progress;
}

sll Cuda::getChecksum()
{
	if (!m_init || !m_kr.d_checksum)
		return 0;

	sll checksum = 0;

	cudaError_t err;
	err = cudaStreamSynchronize(m_kr.s_comp);
	if (err != cudaSuccess) {
		setError("Failed to synchronize compute stream", err);
		return 0;
	}

	DH_COPY(&checksum, m_kr.d_checksum, 1);
	return checksum;
}

const char *Cuda::getDeviceError() const
{
	if (strcmp(m_d_err, "no error") == 0)
		return "No errors in GPU";

	return m_d_err;
}

bool Cuda::getMemoryStats(size_t *free, size_t *total)
{
	if (!m_init)
		return false;

	cudaError_t err = cudaMemGetInfo(free, total);
	CUDA_ERROR("Failed to get memory information");

	return true;
}

bool Cuda::hasEnoughMemory(size_t bytes)
{
	size_t free = 0;
	size_t total = 0;

	if (!getMemoryStats(&free, &total))
		return false;

	const size_t required_memory = bytes;
	if (free < required_memory * 4 / 3) {
		setHostError("Not enough memory for results");
		return false;
	}

	return true;
}
