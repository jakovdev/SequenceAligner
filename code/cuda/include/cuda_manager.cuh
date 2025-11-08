#pragma once
#ifndef CUDA_MANAGER_HPP
#define CUDA_MANAGER_HPP

#include "host_types.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define R __restrict__

struct Sequences {
	Sequences() = default;
	char *d_letters{ nullptr };
	u32 *d_offsets{ nullptr };
	u32 *d_lengths{ nullptr };
	u64 *d_indices{ nullptr };
	u64 n_letters{ 0 };
	u32 n_seqs{ 0 };
	bool constant{ false };
};

struct KernelResults {
	KernelResults() = default;
	~KernelResults()
	{
		cudaFree(d_scores0);
		if (use_batching)
			cudaFree(d_scores1);

		cudaFree(d_progress);
		cudaFree(d_checksum);

		if (s_comp != nullptr)
			cudaStreamDestroy(s_comp);
		if (s_copy != nullptr)
			cudaStreamDestroy(s_copy);
	}

	s32 *d_scores0{ nullptr };
	s32 *d_scores1{ nullptr };
	ull *d_progress{ nullptr };
	sll *d_checksum{ nullptr };
	cudaStream_t s_comp{ nullptr };
	cudaStream_t s_copy{ nullptr };
	s32 *h_scores{ nullptr };
	u64 *h_indices{ nullptr };
	u64 h_batch_size{ 0 };
	u64 h_last_batch{ 0 };
	u64 h_total_count{ 0 };
	u64 h_completed_batch{ 0 };
	ull h_progress{ 0 };
	int h_active{ 0 };
	bool use_batching{ false };
	bool h_after_first{ false };
	bool d_triangular{ false };
	bool copy_in_progress{ false };
};

class Cuda {
    public:
	static Cuda &getInstance();
	Cuda(const Cuda &) = delete;
	Cuda &operator=(const Cuda &) = delete;
	Cuda(Cuda &&) = delete;
	Cuda &operator=(Cuda &&) = delete;
	~Cuda()
	{
		if (!m_init)
			return;

		cudaFree(m_seqs.d_letters);
		cudaFree(m_seqs.d_offsets);
		cudaFree(m_seqs.d_lengths);
		cudaFree(m_seqs.d_indices);
		cudaDeviceReset();
	}

	bool initialize();

	bool hasEnoughMemory(size_t bytes);

	bool uploadSequences(char *sequences_letters, u32 *sequences_offsets,
			     u32 *sequences_lengths, u32 sequences_count,
			     u64 total_sequences_length);

	bool uploadScoring(int *sub_matrix, int *sequence_lookup);
	bool uploadGaps(s32 linear, s32 start, s32 extend);
	bool uploadIndices(u64 *indices, s32 *scores, size_t scores_bytes);

	bool launchKernel(int kernel_id);
	bool getResults();

	ull getProgress();
	sll getChecksum();

	const char *getDeviceError() const;

	const char *getHostError() const
	{
		return m_h_err;
	}

	const char *getDeviceName() const
	{
		return m_init ? m_dev.name : nullptr;
	}

    private:
	Cuda() = default;

	void setHostError(const char *error)
	{
		m_h_err = error;
	}

	void setDeviceError(cudaError_t error)
	{
		m_d_err = cudaGetErrorString(error);
	}

	void setError(const char *host_error, cudaError_t cuda_error)
	{
		setHostError(host_error);
		setDeviceError(cuda_error);
	}

	bool getMemoryStats(size_t *free, size_t *total);
	bool copyTriangularMatrixFlag(bool triangular);

	bool switchKernel(int kernel_id);

	cudaDeviceProp m_dev;
	KernelResults m_kr;
	Sequences m_seqs;
	const char *m_h_err{ "No errors in program" };
	const char *m_d_err{ "No errors from GPU" };
	int m_id{ 0 };
	bool m_init{ false };
};

#define CUDA_ERROR(msg)                     \
	do {                                \
		if (err != cudaSuccess) {   \
			setError(msg, err); \
			return false;       \
		}                           \
	} while (0)

#define D_MEMSET(ptr, value, n)                                 \
	do {                                                    \
		err = cudaMemset(ptr, value, n * sizeof(*ptr)); \
		CUDA_ERROR("D_MEMSET: " #ptr " -> " #value);    \
	} while (0)

#define D_MALLOC(ptr, n)                                  \
	do {                                              \
		err = cudaMalloc(&ptr, n * sizeof(*ptr)); \
		CUDA_ERROR("D_MALLOC: " #ptr);            \
	} while (0)

#define HD_COPY(dst, src, n)                                 \
	do {                                                 \
		static_assert(sizeof(*dst) == sizeof(*src),  \
			      "Pointer types must match");   \
		err = cudaMemcpy(dst, src, n * sizeof(*dst), \
				 cudaMemcpyHostToDevice);    \
		CUDA_ERROR("HD_COPY: " #src " -> " #dst);    \
	} while (0)

#define DH_COPY(dst, src, n)                                 \
	do {                                                 \
		static_assert(sizeof(*dst) == sizeof(*src),  \
			      "Pointer types must match");   \
		err = cudaMemcpy(dst, src, n * sizeof(*dst), \
				 cudaMemcpyDeviceToHost);    \
		CUDA_ERROR("DH_COPY: " #src " -> " #dst);    \
	} while (0)

#define DD_COPY(dst, src, n)                                 \
	do {                                                 \
		static_assert(sizeof(*dst) == sizeof(*src),  \
			      "Pointer types must match");   \
		err = cudaMemcpy(dst, src, n * sizeof(*dst), \
				 cudaMemcpyDeviceToDevice);  \
		CUDA_ERROR("DD_COPY: " #src " -> " #dst);    \
	} while (0)

#define C_COPY(dst, src, n)                              \
	do {                                             \
		err = cudaMemcpyToSymbol(dst, src, n);   \
		CUDA_ERROR("C_COPY: " #src " -> " #dst); \
	} while (0)

#define DH_COPY_ASYNC(dst, src, n, stream)                                    \
	do {                                                                  \
		static_assert(sizeof(*dst) == sizeof(*src),                   \
			      "Pointer types must match");                    \
		err = cudaMemcpyAsync(dst, src, n * sizeof(*dst),             \
				      cudaMemcpyDeviceToHost, stream);        \
		CUDA_ERROR("DH_COPY_ASYNC: " #src " -> " #dst " @ " #stream); \
	} while (0)

#endif // CUDA_MANAGER_HPP
