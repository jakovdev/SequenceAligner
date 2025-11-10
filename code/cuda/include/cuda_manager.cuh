#pragma once
#ifndef CUDA_MANAGER_CUH
#define CUDA_MANAGER_CUH

#include "host_types.h"

#define R __restrict__

struct Sequences {
	Sequences() = default;
	char *d_letters{ nullptr };
	u32 *d_lengths{ nullptr };
	u64 *d_offsets{ nullptr };
	u64 *d_indices{ nullptr };
	u64 n_letters{ 0 };
	u32 n_seqs{ 0 };
	bool constant{ false };
};

struct KernelResults {
	KernelResults() = default;
	s32 *d_scores0{ nullptr };
	s32 *d_scores1{ nullptr };
	ull *d_progress{ nullptr };
	sll *d_checksum{ nullptr };
	cudaStream_t s_comp{ nullptr };
	cudaStream_t s_copy{ nullptr };
	s32 *h_scores{ nullptr };
	u64 h_batch{ 0 };
	u64 h_batch_last{ 0 };
	u64 h_batch_done{ 0 };
	u64 h_alignments{ 0 };
	ull h_progress{ 0 };
	int h_active{ 0 };
	bool use_batching{ false };
	bool h_after_first{ false };
	bool d_triangular{ false };
	bool copy_in_progress{ false };
};

constexpr u64 CUDA_BATCH = (UINT64_C(64) << 20);

class Cuda {
    public:
	Cuda(const Cuda &) = delete;
	Cuda &operator=(const Cuda &) = delete;
	Cuda(Cuda &&) = delete;
	Cuda &operator=(Cuda &&) = delete;
	static Cuda &Instance();
	~Cuda();

	bool initialize();

	bool memoryCheck(size_t bytes);

	bool uploadSequences(const sequence_t *seqs, u32 seq_n,
			     u64 seq_len_sum);
	bool uploadScoring(const s32 sub_mat[SUB_MATDIM][SUB_MATDIM],
			   const s32 seq_lup[SEQ_LUPSIZ]);
	bool uploadGaps(s32 linear, s32 start, s32 extend);
	bool uploadStorage(s32 *scores, size_t scores_bytes);

	bool kernelLaunch(int kernel_id);
	bool kernelResults();
	ull kernelProgress();
	sll kernelChecksum();

	const char *hostError() const;
	const char *deviceError() const;
	const char *deviceName() const;

    private:
	Cuda()
	{
		initialize();
	}

	void hostError(const char *error);
	void deviceError(cudaError_t error);
	void error(const char *host_error, cudaError_t cuda_error)
	{
		hostError(host_error);
		deviceError(cuda_error);
	}

	bool memoryQuery(size_t *free, size_t *total);

	bool copyTriangularMatrixFlag(bool triangular);

	KernelResults m_kr;
	Sequences m_seqs;
	const char *m_h_err{ "No errors in program" };
	const char *m_d_err{ "No errors from GPU" };
	uint m_block_dim{ 0 };
	uint m_grid_size_max{ 0 };
	bool m_init{ false };
	char m_device_name[256]{ "Unknown Device" };
};

#define CUDA_ERROR(msg)                   \
	do {                              \
		if (err != cudaSuccess) { \
			error(msg, err);  \
			return false;     \
		}                         \
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

#endif // CUDA_MANAGER_CUH
