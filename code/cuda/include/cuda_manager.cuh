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

#define TOSTR(x) _TOSTR(x)
#define _TOSTR(x) #x
#define _LINE TOSTR(__LINE__)
#define _FILE TOSTR(__FILE__)
#define MSIZ(p, n) (n * sizeof(*(p)))
#define TO(p, v) p " -> " v
#define N(p, n) p " * " n
#define S(p, s) p " ~ " s

#define ASSERT_PTR_SIZES(a, b) \
	static_assert(sizeof(*(a)) == sizeof(*(b)), "Pointer size mismatch")

#define CUDA_ERROR(msg)                   \
	do {                              \
		if (err != cudaSuccess) { \
			error(msg, err);  \
			return false;     \
		}                         \
	} while (0)

#define CUDA(Func, MACRO, error_message_lit)                                \
	do {                                                                \
		err = cuda##Func;                                           \
		CUDA_ERROR(#MACRO ": " error_message_lit " @ line " _LINE); \
	} while (0)

#define D_MEMSET(p, v, n) \
	CUDA(Memset(p, v, MSIZ(p, n)), D_MEMSET, N(TO(#p, #v), #n))

#define D_MALLOC(p, n) CUDA(Malloc(&p, MSIZ(p, n)), D_MALLOC, N(#p, #n))

#define HD_COPY(d, h, n)                                                \
	ASSERT_PTR_SIZES(d, h);                                         \
	CUDA(Memcpy(d, h, MSIZ(d, n), cudaMemcpyHostToDevice), HD_COPY, \
	     TO(#h, #d))

#define DH_COPY(h, d, n)                                                \
	ASSERT_PTR_SIZES(h, d);                                         \
	CUDA(Memcpy(h, d, MSIZ(h, n), cudaMemcpyDeviceToHost), DH_COPY, \
	     TO(#d, #h))

#define DD_COPY(d, s, n)                                                  \
	ASSERT_PTR_SIZES(d, s);                                           \
	CUDA(Memcpy(d, s, MSIZ(d, n), cudaMemcpyDeviceToDevice), DD_COPY, \
	     TO(#s, #d))

#define C_COPY(d, h) CUDA(MemcpyToSymbol(d, h, sizeof(d)), C_COPY, TO(#h, #d))

#define DH_ACOPY(h, d, n, s)                                           \
	ASSERT_PTR_SIZES(h, d);                                        \
	CUDA(MemcpyAsync(h, d, MSIZ(h, n), cudaMemcpyDeviceToHost, s), \
	     DH_ACOPY, S(TO(#d, #h), #s))

#define D_SYNC(msg_lit) CUDA(DeviceSynchronize(), D_SYNC, msg_lit)

#define S_CREATE(s) CUDA(StreamCreate(&s), S_CREATE, #s)

#define S_SYNC(s) CUDA(StreamSynchronize(s), S_SYNC, #s)

#define S_QUERY(s)           \
	CUDA(StreamQuery(s); \
	     if (err == cudaErrorNotReady) return true, S_QUERY, #s)

#endif // CUDA_MANAGER_CUH
