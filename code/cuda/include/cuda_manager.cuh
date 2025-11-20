#pragma once
#ifndef CUDA_MANAGER_CUH
#define CUDA_MANAGER_CUH

#include "host_types.h"

#define R __restrict__

struct Constants {
	Constants() = default;
	char *letters{ nullptr };
	u32 *lengths{ nullptr };
	u64 *offsets{ nullptr };
	u64 *indices{ nullptr };
	ull *progress{ nullptr };
	sll *checksum{ nullptr };
	s32 sub_mat[SUB_MATSIZE]{ 0 };
	s32 seq_lup[SEQ_LUPSIZ]{ 0 };
	u32 seqs_n{ 0 };
	s32 gap_pen{ 0 };
	s32 gap_open{ 0 };
	s32 gap_ext{ 0 };
	bool triangular{ false };
};

struct Device : Constants {
	Device() = default;
	const char *err{ "No errors from Device" };
	s32 *scores[2]{ nullptr, nullptr };
	cudaStream_t s_comp{ nullptr };
	cudaStream_t s_copy{ nullptr };
	uint bdim{ 0 };
	uint gdim_max{ 0 };
	bool s_sync{ false };
	char name[256]{ "Unknown Device" };

	s32 *current() const noexcept
	{
		return scores[m_active];
	}

	s32 *next() const noexcept
	{
		return scores[1 - m_active];
	}

	void swap() noexcept
	{
		m_active = 1 - m_active;
	}

    private:
	int m_active{ 0 };
};

struct States {
	States() = default;
	bool init{ false };
	bool seqs{ false };
	bool scoring{ false };
	bool gaps{ false };
	bool storage{ false };

	bool ready() const noexcept
	{
		return seqs && scoring && gaps && storage;
	}
};

struct Host {
	const char *err{ "No errors from Host" };
	s32 *scores{ nullptr };
	u64 batch{ 0 };
	u64 batch_last{ 0 };
	u64 batch_done{ 0 };
	u64 alignments{ 0 };
	ull progress{ 0 };
	bool subsequent{ false };
};

class Cuda {
    public:
	Cuda(const Cuda &) = delete;
	Cuda &operator=(const Cuda &) = delete;
	Cuda(Cuda &&) = delete;
	Cuda &operator=(Cuda &&) = delete;
	static Cuda &Instance();
	~Cuda() noexcept;

	bool memoryCheck(size_t bytes) noexcept;

	bool uploadSequences(const sequence_t *seqs, u32 seq_n, u32 seq_len_max,
			     u64 seq_len_sum) noexcept;
	bool uploadScoring(const s32 sub_mat[SUB_MATDIM][SUB_MATDIM],
			   const s32 seq_lup[SEQ_LUPSIZ]) noexcept;
	bool uploadGaps(s32 linear, s32 start, s32 extend) noexcept;
	bool uploadStorage(s32 *scores, size_t scores_bytes) noexcept;

	bool kernelLaunch(int kernel_id) noexcept;
	bool kernelResults() noexcept;
	ull kernelProgress() const noexcept;
	sll kernelChecksum() noexcept;

	const char *hostError() const noexcept;
	const char *deviceError() const noexcept;
	const char *deviceName() const noexcept;

    private:
	Cuda() noexcept;

	void hostError(const char *error) noexcept;
	void deviceError(cudaError_t error) noexcept;

	bool memoryQuery(size_t *free, size_t *total) noexcept;

	Device d;
	Host h;
	States s;
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

#define CUDA_ERROR(msg, retval)           \
	do {                              \
		if (err != cudaSuccess) { \
			deviceError(err); \
			hostError(msg);   \
			return retval;    \
		}                         \
	} while (0)

#define CUDA(Func, MACRO, error_msg_lit)                                       \
	do {                                                                   \
		err = cuda##Func;                                              \
		CUDA_ERROR(#MACRO ": " error_msg_lit " @ line " _LINE, false); \
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

#endif /* CUDA_MANAGER_CUH */
