#include "cuda_manager.cuh"
#include "host_types.h"

constexpr u64 CUDA_BATCH_SIZE = (UINT64_C(64) << 20);

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

bool Cuda::uploadSequences(char *sequences_letters, u32 *sequences_offsets,
			   u32 *sequences_lengths, u32 sequences_count,
			   u64 total_sequences_length)
{
	if (!m_init || !sequences_letters || !sequences_offsets ||
	    !sequences_lengths || !sequences_count || !total_sequences_length) {
		setHostError("Invalid parameters for sequence upload");
		return false;
	}

	m_seqs.n_seqs = sequences_count;
	m_seqs.n_letters = total_sequences_length;
	m_kr.h_total_count = (u64)sequences_count * (sequences_count - 1) / 2;
	constexpr size_t cell_size = sizeof(*m_kr.h_scores);

	if (!hasEnoughMemory(cell_size * sequences_count * sequences_count)) {
		if (!hasEnoughMemory(cell_size * m_kr.h_total_count)) {
			if (!hasEnoughMemory(cell_size * CUDA_BATCH_SIZE))
				return false;

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

bool Cuda::uploadIndices(u64 *indices, s32 *scores, size_t scores_bytes)
{
	if (!m_init || !indices || !scores || !scores_bytes) {
		setHostError(
			"Invalid parameters for uploading indices, or using --no-write");
		return false;
	}

	const size_t expected_size = m_kr.h_total_count * sizeof(*scores);

	if (scores_bytes <= expected_size) {
		if (scores_bytes == expected_size) {
			if (!copyTriangularMatrixFlag(true))
				return false;
		} else {
			setHostError("Buffer size is too small for results");
			return false;
		}
	}

	m_kr.h_scores = scores;
	m_kr.h_indices = indices;

	cudaError_t err;
	D_MALLOC(m_seqs.d_indices, m_seqs.n_seqs);
	HD_COPY(m_seqs.d_indices, indices, m_seqs.n_seqs);

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
		u64 d_scores_size = 0;

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

			const u64 n_scores = m_seqs.n_seqs * m_seqs.n_seqs;
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

		CUDA_ERROR("Copy stream query failed");
		m_kr.copy_in_progress = false;
	}

	if (!m_kr.h_after_first && m_kr.h_batch_size < m_kr.h_total_count) {
		m_kr.h_after_first = true;
		return true;
	}

	u64 batch_offset = m_kr.h_completed_batch;
	s32 *buffer = nullptr;

	if (m_kr.h_after_first) {
		buffer = (m_kr.h_active == 0) ? m_kr.d_scores1 : m_kr.d_scores0;
	} else {
		err = cudaStreamSynchronize(m_kr.s_comp);
		CUDA_ERROR("Compute stream synchronization failed");
		DH_COPY(&m_kr.h_progress, m_kr.d_progress, 1);
		buffer = m_kr.d_scores0;
	}

	u64 n_scores =
		std::min(m_kr.h_batch_size, m_kr.h_total_count - batch_offset);

	if (n_scores > 0) {
		s32 *scores = m_kr.h_scores + batch_offset;

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
	CUDA_ERROR("Failed to synchronize compute stream");

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
