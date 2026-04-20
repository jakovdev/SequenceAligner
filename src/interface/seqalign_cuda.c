#include "interface/seqalign_cuda.h"

#include <args.h>
#include <print.h>

#ifdef USE_CUDA
#ifdef __MINGW64__
#undef __cdecl
#endif
#include <cuda_runtime_api.h>
#include <string.h>

#include "bio/kernels.cuh"
#include "bio/sequences.h"
#include "interface/seqalign_hdf5.h"
#include "util/benchmark.h"
#include "util/macros.h"

#define CALLR(cuda_func)                                     \
	do {                                                 \
		err = cuda_func;                             \
		if unlikely (err != cudaSuccess) {           \
			perr("%s", cudaGetErrorString(err)); \
			return false;                        \
		}                                            \
	} while (0)

#define CALLJ(cuda_func, jmp_label)                          \
	do {                                                 \
		err = cuda_func;                             \
		if unlikely (err != cudaSuccess) {           \
			perr("%s", cudaGetErrorString(err)); \
			goto jmp_label;                      \
		}                                            \
	} while (0)

static void cuda_device_close(void)
{
	cudaDeviceReset();
}

static bool cuda_device_init(void)
{
	static bool init;
	if (init)
		return true;

	cudaError_t err = {};

	int device_count = 0;
	err = cudaGetDeviceCount(&device_count);
	if (!device_count || err != cudaSuccess) {
		perr("No CUDA devices available");
		return false;
	}

	CALLR(cudaSetDevice(0));
	atexit(cuda_device_close);
	init = true;
	return true;
}

static bool no_cuda;

bool cuda_memory(size_t bytes)
{
	if (no_cuda)
		return true;

	if (!cuda_device_init())
		return false;

	size_t free = 0;
	size_t total = 0;
	cudaError_t err = {};
	CALLJ(cudaMemGetInfo(&free, &total), memory_error);
	if (free < bytes * 4 / 3)
		return false;

	return true;
memory_error:
	exit(EXIT_FAILURE);
}

bool cuda_align(const struct input *dataset)
{
	if (no_cuda)
		return align(dataset);

	if (!cuda_device_init())
		return false;

	if (dataset->lengths_max > MAX_CUDA_SEQUENCE_LENGTH) {
		perr("Sequence length exceeds CUDA Device limits");
		return false;
	}

	cudaError_t err = {};
	uint block_max = 0;
	{
		struct cudaDeviceProp dev_prop = {};
		CALLR(cudaGetDeviceProperties(&dev_prop, 0));
		pinfo("Using CUDA device: %s", dev_prop.name);
		block_max = (uint)dev_prop.maxThreadsPerBlock;
	}

	struct Constants C = {
		.seq_n = dataset->seqs_n,
		.gap_pen = GAP_PEN,
		.gap_open = GAP_OPN,
		.gap_ext = GAP_EXT,
	};

	memcpy(C.seq_lut, SEQ_LUT, sizeof(C.seq_lut));
	memcpy(C.sub_mat, SUB_MAT, sizeof(C.sub_mat));

	size_t seq_n = (size_t)dataset->seqs_n;
	s64 *OFFSETS = dataset->offsets;
	s32 *LENGTHS = dataset->lengths;
	char *LETTERS = dataset->letters;
	size_t sum = (size_t)(OFFSETS[seq_n - 1] + LENGTHS[seq_n - 1] + 1);

	CALLR(cudaMalloc((void **)&C.offsets, sizeof(*C.offsets) * seq_n));
	CALLR(cudaMalloc((void **)&C.lengths, sizeof(*C.lengths) * seq_n));
	CALLR(cudaMalloc((void **)&C.letters, sizeof(*C.letters) * sum));

	CALLR(cudaMemcpy(C.offsets, OFFSETS, sizeof(*C.offsets) * seq_n,
			 cudaMemcpyHostToDevice));
	CALLR(cudaMemcpy(C.lengths, LENGTHS, sizeof(*C.lengths) * seq_n,
			 cudaMemcpyHostToDevice));
	CALLR(cudaMemcpy(C.letters, LETTERS, sizeof(*C.letters) * sum,
			 cudaMemcpyHostToDevice));

	const s64 alignments = dataset->alignments;
	const s64 batch_size = INT64_C(64) << 20;
	s32 *matrix = h5_matrix_data();

	if (!cuda_memory(bytesof(matrix, seq_n * seq_n))) {
		if (!cuda_memory(bytesof(matrix, (size_t)alignments))) {
			if (!cuda_memory(bytesof(matrix, (size_t)batch_size))) {
				perr("Not enough CUDA Device memory for alignment");
				return false;
			}
		}
		C.triangular = true;
	}

	if (h5_matrix_bytes() == (sizeof(*matrix) * (size_t)alignments))
		C.triangular = true;

	s64 batch = 0, batch_last = 0, batch_done = 0;
	s32 *scores[2] = {};
	s32 active = 0;
	if (C.triangular) {
		batch = min(batch_size, alignments);
		CALLR(cudaMalloc((void **)&scores[0],
				 sizeof(*scores[0]) * (size_t)batch));
		CALLR(cudaMemset(scores[0], 0,
				 sizeof(*scores[0]) * (size_t)batch));
		CALLR(cudaMalloc((void **)&scores[1],
				 sizeof(*scores[1]) * (size_t)batch));
		CALLR(cudaMemset(scores[1], 0,
				 sizeof(*scores[1]) * (size_t)batch));
	} else {
		batch = alignments;
		CALLR(cudaMalloc((void **)&scores[0],
				 sizeof(*scores[0]) * seq_n * seq_n));
		CALLR(cudaMemset(scores[0], 0,
				 sizeof(*scores[0]) * seq_n * seq_n));
	}

	CALLR(cudaMalloc((void **)&C.progress, sizeof(*C.progress)));
	CALLR(cudaMalloc((void **)&C.checksum, sizeof(*C.checksum)));
	CALLR(cudaMemset(C.progress, 0, sizeof(*C.progress)));
	CALLR(cudaMemset(C.checksum, 0, sizeof(*C.checksum)));
	CALLR(copy_constants(&C));

	const void *kernel = kernels[METHOD_ID];
	dim3 block = { block_max, 1, 1 };
	cudaStream_t compute = {}, memory = {};
	CALLR(cudaStreamCreate(&compute));
	CALLR(cudaStreamCreate(&memory));

	bool subsequent = false, syncing = false, matrix_copied = false;
	s64 progress = 0;

	pinfol("Performing %w64d pairwise alignments", alignments);

	ppercent(0, "Aligning sequences");
	bench_align_start();
	while (true) {
		s64 offset = batch_last;
		if (offset >= alignments) {
			if (subsequent) {
				CALLR(cudaDeviceSynchronize());
				CALLR(cudaMemcpy(&progress, C.progress,
						 sizeof(progress),
						 cudaMemcpyDeviceToHost));
				active = 1 - active;
			}
			goto cuda_results;
		}

		if (C.triangular) {
			if (offset + batch > alignments)
				batch = alignments - offset;
			if (!batch) {
				if (subsequent) {
					CALLR(cudaDeviceSynchronize());
					CALLR(cudaMemcpy(
						&progress, C.progress,
						sizeof(progress),
						cudaMemcpyDeviceToHost));
				}
				goto cuda_results;
			}
			if (subsequent) {
				CALLR(cudaDeviceSynchronize());
				CALLR(cudaMemcpy(&progress, C.progress,
						 sizeof(progress),
						 cudaMemcpyDeviceToHost));
				active = 1 - active;
			}
		}

		dim3 grid = { (uint)((batch + block.x - 1) / block.x), 1, 1 };
		void *args[] = { &scores[active], &offset, &batch };
		CALLR(cudaLaunchKernel(kernel, grid, block, args, 0, compute));
		batch_last += batch;
cuda_results:

		if (!C.triangular) {
			if (matrix_copied)
				goto cuda_progress;

			CALLR(cudaStreamSynchronize(compute));
			CALLR(cudaMemcpy(&progress, C.progress,
					 sizeof(progress),
					 cudaMemcpyDeviceToHost));

			if (matrix)
				CALLR(cudaMemcpy(matrix, scores[0],
						 sizeof(*matrix) * seq_n *
							 seq_n,
						 cudaMemcpyDeviceToHost));

			matrix_copied = true;
			goto cuda_progress;
		}

		if (batch_done >= alignments) {
			if (syncing) {
				CALLR(cudaStreamSynchronize(memory));
				syncing = false;
			}
			goto cuda_progress;
		}

		if (syncing) {
			err = cudaStreamQuery(memory);
			if (err == cudaErrorNotReady)
				goto cuda_progress;
			CALLR(err);
			syncing = false;
		}

		if (!subsequent && batch < alignments) {
			subsequent = true;
			goto cuda_progress;
		}

		size_t n_scores = (size_t)min(batch, alignments - batch_done);
		if (!n_scores)
			goto cuda_progress;

		if (subsequent) {
			if (matrix)
				CALLR(cudaMemcpyAsync(
					matrix + batch_done, scores[1 - active],
					sizeof(*matrix) * n_scores,
					cudaMemcpyDeviceToHost, memory));
			syncing = true;
		} else {
			CALLR(cudaStreamSynchronize(compute));
			CALLR(cudaMemcpy(&progress, C.progress,
					 sizeof(progress),
					 cudaMemcpyDeviceToHost));
			if (matrix)
				CALLR(cudaMemcpy(matrix + batch_done,
						 scores[active],
						 sizeof(*matrix) * n_scores,
						 cudaMemcpyDeviceToHost));
		}
		batch_done += (s64)n_scores;
cuda_progress:
		pproportc(progress / alignments, "Aligning sequences");
		if (progress >= alignments)
			break;
	}

	bench_align_end();
	ppercent(100, "Aligning sequences");

	sll checksum = 0;
	CALLR(cudaMemcpy(&checksum, C.checksum, sizeof(checksum),
			 cudaMemcpyDeviceToHost));
	h5_checksum_set(checksum * 2);

	bench_align_print();
	return true;
}

static void print_no_cuda(void)

{
	pinfom("CUDA: Enabled");
}

ARG_EXTERN(compression);
ARG_EXTERN(threads);

ARGUMENT(disable_cuda) = {
	.opt = 'C',
	.lopt = "no-cuda",
	.help = "Disable CUDA",
	.set = &no_cuda,
	.action_callback = print_no_cuda,
	.action_phase = ARG_CALLBACK_IF_UNSET,
	.action_order = ARG_ORDER_AFTER(ARG(compression)),
	.help_order = ARG_ORDER_AFTER(ARG(threads)),
};

#else

#include "bio/alignment.h"

bool cuda_memory(size_t)
{
	return true;
}

bool cuda_align(const struct input *dataset)
{
	return align(dataset);
}

static void print_cuda_ignored(void)
{
	pwarnm("CUDA: Ignored");
}

ARG_EXTERN(compression);

ARGUMENT(disable_cuda) = {
	.opt = 'C',
	.lopt = "no-cuda",
	.arg_req = ARG_HIDDEN,
	.action_callback = print_cuda_ignored,
	.action_phase = ARG_CALLBACK_IF_SET,
	.action_order = ARG_ORDER_AFTER(ARG(compression)),
};

#endif /* USE_CUDA */
