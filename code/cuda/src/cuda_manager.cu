#include "cuda_manager.cuh"
#include "host_types.h"
#include <memory>

constexpr u64 CUDA_BATCH = (UINT64_C(64) << 20);

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

bool Cuda::uploadSequences(sequence_t *seqs, u32 seq_n, u64 seq_len_total)
{
	if (!m_init || !seqs || seq_n < SEQUENCE_COUNT_MIN) {
		setHostError("Invalid parameters for sequence upload");
		return false;
	}

	m_seqs.n_seqs = seq_n;
	m_seqs.n_letters = seq_len_total;
	m_kr.h_alignments = (u64)seq_n * (seq_n - 1) / 2;
	constexpr size_t cell_size = sizeof(*m_kr.h_scores);

	if (!hasEnoughMemory(cell_size * seq_n * seq_n)) {
		if (!hasEnoughMemory(cell_size * m_kr.h_alignments)) {
			if (!hasEnoughMemory(cell_size * CUDA_BATCH))
				return false;

			m_kr.use_batching = true;
		}

		if (!copyTriangularMatrixFlag(true))
			return false;

		setHostError("No errors in program");
	}

	auto offs_flat = std::make_unique<u32[]>(seq_n);
	auto lens_flat = std::make_unique<u32[]>(seq_n);
	auto seqs_flat = std::make_unique<char[]>(seq_len_total);

	u32 offs = 0;
	for (u32 i = 0; i < seq_n; i++) {
		offs_flat[i] = offs;
		lens_flat[i] = static_cast<u32>(seqs[i].length);
		memcpy(seqs_flat.get() + offs, seqs[i].letters, seqs[i].length);
		offs += static_cast<u32>(seqs[i].length);
	}

	cudaError_t err;

	D_MALLOC(m_seqs.d_offsets, seq_n);
	D_MALLOC(m_seqs.d_lengths, seq_n);
	D_MALLOC(m_seqs.d_letters, seq_len_total);

	HD_COPY(m_seqs.d_offsets, offs_flat.get(), seq_n);
	HD_COPY(m_seqs.d_lengths, lens_flat.get(), seq_n);
	HD_COPY(m_seqs.d_letters, seqs_flat.get(), seq_len_total);

	return true;
}

bool Cuda::uploadIndices(u64 *indices, s32 *scores, size_t scores_bytes)
{
	if (!m_init || !indices || !scores || !scores_bytes) {
		setHostError(
			"Invalid parameters for uploading indices, or using --no-write");
		return false;
	}

	const size_t expected_size = m_kr.h_alignments * sizeof(*scores);

	if (scores_bytes < expected_size) {
		setHostError("Buffer size is too small for results");
		return false;
	}

	if (scores_bytes == expected_size && !copyTriangularMatrixFlag(true))
		return false;

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
			m_kr.h_batch = std::min(CUDA_BATCH, m_kr.h_alignments);
			d_scores_size = m_kr.h_batch;

			D_MALLOC(m_kr.d_scores0, d_scores_size);
			D_MALLOC(m_kr.d_scores1, d_scores_size);
		} else {
			m_kr.h_batch = m_kr.h_alignments;
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
		if (matrix_copied)
			return true;

		err = cudaStreamSynchronize(m_kr.s_comp);
		CUDA_ERROR("Compute stream synchronization failed");
		DH_COPY(&m_kr.h_progress, m_kr.d_progress, 1);

		const u64 n_scores = m_seqs.n_seqs * m_seqs.n_seqs;
		DH_COPY(m_kr.h_scores, m_kr.d_scores0, n_scores);
		matrix_copied = true;
		return true;
	}

	if (m_kr.h_batch_done >= m_kr.h_alignments) {
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

	if (!m_kr.h_after_first && m_kr.h_batch < m_kr.h_alignments) {
		m_kr.h_after_first = true;
		return true;
	}

	u64 batch_offset = m_kr.h_batch_done;
	s32 *buffer = nullptr;

	if (m_kr.h_after_first) {
		buffer = (m_kr.h_active == 0) ? m_kr.d_scores1 : m_kr.d_scores0;
	} else {
		err = cudaStreamSynchronize(m_kr.s_comp);
		CUDA_ERROR("Compute stream synchronization failed");
		DH_COPY(&m_kr.h_progress, m_kr.d_progress, 1);
		buffer = m_kr.d_scores0;
	}

	u64 n_scores = std::min(m_kr.h_batch, m_kr.h_alignments - batch_offset);
	if (!n_scores)
		return true;

	s32 *scores = m_kr.h_scores + batch_offset;

	if (m_kr.h_after_first) {
		DH_COPY_ASYNC(scores, buffer, n_scores, m_kr.s_copy);
		m_kr.copy_in_progress = true;
	} else {
		DH_COPY(scores, buffer, n_scores);
	}

	m_kr.h_batch_done += n_scores;
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
