#include "cuda_manager.cuh"

#include <vector>
#include <cstring>

Cuda &Cuda::getInstance()
{
	static Cuda instance;
	return instance;
}

bool Cuda::initialize()
{
	if (m_init)
		return true;

	int device_count = 0;
	cudaError_t err = cudaGetDeviceCount(&device_count);
	CUDA_ERROR("Failed to get device count or none available");

	if (!device_count) {
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

bool Cuda::uploadSequences(const sequence_t *seqs, u32 seq_n, u64 seq_len_sum)
{
	if (!m_init || !seqs || seq_n < SEQUENCE_COUNT_MIN) {
		setHostError("Invalid parameters for sequence upload");
		return false;
	}

	m_seqs.n_seqs = seq_n;
	m_seqs.n_letters = seq_len_sum;
	m_kr.h_alignments = static_cast<u64>(seq_n) * (seq_n - 1) / 2;
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

	std::vector<u64> indices_f(seq_n);
	std::vector<u64> offsets_f(seq_n);
	std::vector<u32> lengths_f(seq_n);
	std::vector<char> letters_f(seq_len_sum);

	auto indices_p = indices_f.data();
	auto offsets_p = offsets_f.data();
	auto lengths_p = lengths_f.data();
	auto letters_p = letters_f.data();

	for (u64 i = 0, offs = 0; i < seq_n; i++) {
		indices_p[i] = (i * (i - 1)) / 2;
		offsets_p[i] = offs;
		lengths_p[i] = static_cast<u32>(seqs[i].length);
		std::memcpy(letters_p + offs, seqs[i].letters, seqs[i].length);
		offs += seqs[i].length;
	}

	cudaError_t err;

	D_MALLOC(m_seqs.d_indices, seq_n);
	D_MALLOC(m_seqs.d_offsets, seq_n);
	D_MALLOC(m_seqs.d_lengths, seq_n);
	D_MALLOC(m_seqs.d_letters, seq_len_sum);

	HD_COPY(m_seqs.d_indices, indices_p, seq_n);
	HD_COPY(m_seqs.d_offsets, offsets_p, seq_n);
	HD_COPY(m_seqs.d_lengths, lengths_p, seq_n);
	HD_COPY(m_seqs.d_letters, letters_p, seq_len_sum);

	return true;
}

bool Cuda::uploadStorage(s32 *scores, size_t scores_bytes)
{
	if (!m_init || !scores || !scores_bytes) {
		setHostError(
			"Invalid parameters for storage upload, or using --no-write");
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

	if (free < bytes * 4 / 3) {
		setHostError("Not enough memory for results");
		return false;
	}

	return true;
}
