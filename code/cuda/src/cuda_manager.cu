#include "cuda_manager.cuh"

#include <vector>
#include <cstring>

Cuda &Cuda::Instance()
{
	static Cuda instance;
	return instance;
}

Cuda::~Cuda() noexcept
{
	if (!s.init)
		return;

	if (s.seqs) {
		cudaFree(d.letters);
		cudaFree(d.offsets);
		cudaFree(d.lengths);
		cudaFree(d.indices);
	}

	if (!s.ready())
		return;

	cudaFree(d.progress);
	cudaFree(d.checksum);

	if (d.s_comp)
		cudaStreamDestroy(d.s_comp);
	if (d.s_copy)
		cudaStreamDestroy(d.s_copy);

	cudaFree(d.scores[0]);
	if (d.triangular)
		cudaFree(d.scores[1]);

	cudaDeviceReset();
}

Cuda::Cuda() noexcept
{
	if (s.init)
		return;

	int device_count = 0;
	cudaError_t err = cudaGetDeviceCount(&device_count);
	CUDA_ERROR("Failed to get device count or none available", );

	if (!device_count) {
		deviceError(err);
		hostError("No CUDA devices available");
		return;
	}

	err = cudaSetDevice(0);
	CUDA_ERROR("Failed to set CUDA device", );

	cudaDeviceProp dev_prop;
	err = cudaGetDeviceProperties(&dev_prop, 0);
	CUDA_ERROR("Failed to get device properties", );
	std::strcpy(d.name, dev_prop.name);
	d.bdim = static_cast<uint>(dev_prop.maxThreadsPerBlock);
	d.gdim_max = static_cast<uint>(dev_prop.maxGridSize[0]);

	cudaGetLastError();
	s.init = true;
}

template <typename T> bool vecalloc(std::vector<T> &v, const size_t n) noexcept
{
	try {
		v.resize(n);
	} catch (const std::bad_alloc &) {
		return false;
	}
	return !v.empty();
}

bool Cuda::uploadSequences(const sequence_t *seqs, u32 seq_n,
			   u64 seq_len_sum) noexcept
{
	if (!s.init || !seqs || seq_n < SEQUENCE_COUNT_MIN) {
		hostError("Invalid parameters for sequence upload");
		return false;
	}

	std::vector<u64> indices_f;
	std::vector<u64> offsets_f;
	std::vector<u32> lengths_f;
	std::vector<char> letters_f;

	if (!vecalloc(indices_f, seq_n) || !vecalloc(offsets_f, seq_n) ||
	    !vecalloc(lengths_f, seq_n) || !vecalloc(letters_f, seq_len_sum)) {
		hostError("Memory allocation failed for sequence upload");
		return false;
	}

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

	D_MALLOC(d.indices, seq_n);
	D_MALLOC(d.offsets, seq_n);
	D_MALLOC(d.lengths, seq_n);
	D_MALLOC(d.letters, seq_len_sum);

	HD_COPY(d.indices, indices_p, seq_n);
	HD_COPY(d.offsets, offsets_p, seq_n);
	HD_COPY(d.lengths, lengths_p, seq_n);
	HD_COPY(d.letters, letters_p, seq_len_sum);

	d.seqs_n = seq_n;
	s.seqs = true;

	return true;
}

bool Cuda::uploadScoring(const s32 sub_mat[SUB_MATDIM][SUB_MATDIM],
			 const s32 seq_lup[SEQ_LUPSIZ]) noexcept
{
	if (!s.init) {
		hostError("CUDA not initialized");
		return false;
	}

	std::memcpy(d.sub_mat, sub_mat, sizeof(d.sub_mat));
	std::memcpy(d.seq_lup, seq_lup, sizeof(d.seq_lup));
	s.scoring = true;

	return true;
}

bool Cuda::uploadGaps(s32 linear, s32 open, s32 extend) noexcept
{
	if (!s.init) {
		hostError("CUDA not initialized");
		return false;
	}

	d.gap_pen = linear;
	d.gap_open = open;
	d.gap_ext = extend;
	s.gaps = true;

	return true;
}

bool Cuda::uploadStorage(s32 *scores, size_t scores_bytes) noexcept
{
	if (!s.init || !s.seqs || !scores || !scores_bytes) {
		hostError("Invalid context, storage, or using --no-write");
		return false;
	}

	h.alignments = static_cast<u64>(d.seqs_n) * (d.seqs_n - 1) / 2;
	constexpr u64 batch_size = UINT64_C(64) << 20;

	if (!memoryCheck(sizeof(*h.scores) * d.seqs_n * d.seqs_n)) {
		if (!memoryCheck(sizeof(*h.scores) * h.alignments)) {
			if (!memoryCheck(sizeof(*h.scores) * batch_size))
				return false;
		}

		d.triangular = true;
		hostError("No errors from Host");
	}

	const size_t expected_size = h.alignments * sizeof(*scores);

	if (scores_bytes < expected_size) {
		hostError("Buffer size is too small for results");
		return false;
	}

	if (scores_bytes == expected_size)
		d.triangular = true;

	cudaError_t err;

	S_CREATE(d.s_comp);
	S_CREATE(d.s_copy);

	if (d.triangular) {
		h.batch = std::min(batch_size, h.alignments);
		D_MALLOC(d.scores[0], h.batch);
		D_MALLOC(d.scores[1], h.batch);
	} else {
		h.batch = h.alignments;
		D_MALLOC(d.scores[0], d.seqs_n * d.seqs_n);
	}

	h.scores = scores;
	s.storage = true;

	return true;
}

const char *Cuda::hostError() const noexcept
{
	return h.err;
}

const char *Cuda::deviceError() const noexcept
{
	if (strcmp(d.err, "no error") == 0)
		return "No errors from Device";

	return d.err;
}

const char *Cuda::deviceName() const noexcept
{
	return d.name;
}

void Cuda::hostError(const char *error) noexcept
{
	h.err = error;
}

void Cuda::deviceError(cudaError_t error) noexcept
{
	d.err = cudaGetErrorString(error);
}

bool Cuda::memoryQuery(size_t *free, size_t *total) noexcept
{
	if (!s.init)
		return false;

	cudaError_t err = cudaMemGetInfo(free, total);
	CUDA_ERROR("Failed to get memory information", false);

	return true;
}

bool Cuda::memoryCheck(size_t bytes) noexcept
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
