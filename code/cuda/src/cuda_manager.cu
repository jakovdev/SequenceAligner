#include "cuda_manager.cuh"

#include <vector>
#include <cstring>

Cuda &Cuda::Instance()
{
	static Cuda instance;
	return instance;
}

Cuda::~Cuda()
{
	if (!m_init)
		return;

	cudaFree(m_seqs.d_letters);
	cudaFree(m_seqs.d_offsets);
	cudaFree(m_seqs.d_lengths);
	cudaFree(m_seqs.d_indices);

	cudaFree(m_kr.d_progress);
	cudaFree(m_kr.d_checksum);
	cudaFree(m_kr.d_scores0);
	if (m_kr.use_batching)
		cudaFree(m_kr.d_scores1);
	if (m_kr.s_comp)
		cudaStreamDestroy(m_kr.s_comp);
	if (m_kr.s_copy)
		cudaStreamDestroy(m_kr.s_copy);
	cudaDeviceReset();
}

bool Cuda::initialize()
{
	if (m_init)
		return true;

	int device_count = 0;
	cudaError_t err = cudaGetDeviceCount(&device_count);
	CUDA_ERROR("Failed to get device count or none available");

	if (!device_count) {
		error("No CUDA devices available", err);
		return false;
	}

	err = cudaSetDevice(0);
	CUDA_ERROR("Failed to set CUDA device");

	cudaDeviceProp dev_prop;
	err = cudaGetDeviceProperties(&dev_prop, 0);
	CUDA_ERROR("Failed to get device properties");
	std::strncpy(m_device_name, dev_prop.name, sizeof(m_device_name));
	m_device_name[sizeof(m_device_name) - 1] = '\0';
	m_block_dim = static_cast<uint>(dev_prop.maxThreadsPerBlock);
	m_grid_size_max = static_cast<uint>(dev_prop.maxGridSize[0]);

	cudaGetLastError();
	m_init = true;

	return true;
}

bool Cuda::uploadSequences(const sequence_t *seqs, u32 seq_n, u64 seq_len_sum)
{
	if (!m_init || !seqs || seq_n < SEQUENCE_COUNT_MIN) {
		hostError("Invalid parameters for sequence upload");
		return false;
	}

	m_seqs.n_seqs = seq_n;
	m_seqs.n_letters = seq_len_sum;
	m_kr.h_alignments = static_cast<u64>(seq_n) * (seq_n - 1) / 2;
	constexpr size_t cell_size = sizeof(*m_kr.h_scores);

	if (!memoryCheck(cell_size * seq_n * seq_n)) {
		if (!memoryCheck(cell_size * m_kr.h_alignments)) {
			if (!memoryCheck(cell_size * CUDA_BATCH))
				return false;

			m_kr.use_batching = true;
		}

		if (!copyTriangularMatrixFlag(true))
			return false;

		hostError("No errors in program");
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
		hostError(
			"Invalid parameters for storage upload, or using --no-write");
		return false;
	}

	const size_t expected_size = m_kr.h_alignments * sizeof(*scores);

	if (scores_bytes < expected_size) {
		hostError("Buffer size is too small for results");
		return false;
	}

	if (scores_bytes == expected_size && !copyTriangularMatrixFlag(true))
		return false;

	m_kr.h_scores = scores;

	return true;
}

bool Cuda::kernelResults()
{
	if (!m_init || !m_kr.h_scores) {
		hostError("Invalid context or results storage");
		return false;
	}

	cudaError_t err;

	if (!m_kr.d_triangular) {
		static bool matrix_copied = false;
		if (matrix_copied)
			return true;

		S_SYNC(m_kr.s_comp);
		DH_COPY(&m_kr.h_progress, m_kr.d_progress, 1);

		const u64 n_scores = m_seqs.n_seqs * m_seqs.n_seqs;
		DH_COPY(m_kr.h_scores, m_kr.d_scores0, n_scores);
		matrix_copied = true;
		return true;
	}

	if (m_kr.h_batch_done >= m_kr.h_alignments) {
		if (m_kr.copy_in_progress) {
			S_SYNC(m_kr.s_copy);
			m_kr.copy_in_progress = false;
		}

		return true;
	}

	if (m_kr.copy_in_progress) {
		S_QUERY(m_kr.s_copy);
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
		S_SYNC(m_kr.s_comp);
		DH_COPY(&m_kr.h_progress, m_kr.d_progress, 1);
		buffer = m_kr.d_scores0;
	}

	u64 n_scores = std::min(m_kr.h_batch, m_kr.h_alignments - batch_offset);
	if (!n_scores)
		return true;

	s32 *scores = m_kr.h_scores + batch_offset;

	if (m_kr.h_after_first) {
		DH_ACOPY(scores, buffer, n_scores, m_kr.s_copy);
		m_kr.copy_in_progress = true;
	} else {
		DH_COPY(scores, buffer, n_scores);
	}

	m_kr.h_batch_done += n_scores;
	return true;
}

ull Cuda::kernelProgress()
{
	if (!m_init)
		return 0;

	return m_kr.h_progress;
}

sll Cuda::kernelChecksum()
{
	if (!m_init || !m_kr.d_checksum)
		return 0;

	cudaError_t err;
	S_SYNC(m_kr.s_comp);

	sll checksum = 0;
	DH_COPY(&checksum, m_kr.d_checksum, 1);
	return checksum;
}

const char *Cuda::hostError() const
{
	return m_h_err;
}

const char *Cuda::deviceError() const
{
	if (strcmp(m_d_err, "no error") == 0)
		return "No errors in GPU";

	return m_d_err;
}

const char *Cuda::deviceName() const
{
	return m_init ? m_device_name : nullptr;
}

void Cuda::hostError(const char *error)
{
	m_h_err = error;
}

void Cuda::deviceError(cudaError_t error)
{
	m_d_err = cudaGetErrorString(error);
}

bool Cuda::memoryQuery(size_t *free, size_t *total)
{
	if (!m_init)
		return false;

	cudaError_t err = cudaMemGetInfo(free, total);
	CUDA_ERROR("Failed to get memory information");

	return true;
}

bool Cuda::memoryCheck(size_t bytes)
{
	size_t free = 0;
	size_t total = 0;

	if (!memoryQuery(&free, &total))
		return false;

	if (free < bytes * 4 / 3) {
		hostError("Not enough memory for results");
		return false;
	}

	return true;
}
