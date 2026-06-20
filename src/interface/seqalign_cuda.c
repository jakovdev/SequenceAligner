#include "interface/seqalign_cuda.h"

#include <args.h>
#include <print.h>

[[gnu::nonnull]]
bool align(struct input, struct output);

#ifdef USE_CUDA
#ifdef __MINGW64__
#undef __cdecl
#endif
#include <cuda_runtime_api.h>
#include <string.h>

#include "bio/kernels.cuh"
#include "io/input.h"
#include "io/output.h"
#include "util/benchmark.h"
#include "util/macros.h"

#define CALLR(cuda_func)                                     \
	do {                                                 \
		err = cuda_func;                             \
		if (err != cudaSuccess) {                    \
			perr("%s", cudaGetErrorString(err)); \
			return false;                        \
		}                                            \
	} while (0)

#define CALLJ(cuda_func, jmp_label)                          \
	do {                                                 \
		err = cuda_func;                             \
		if (err != cudaSuccess) {                    \
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

	int device_count = 0;
	cudaError_t err = cudaGetDeviceCount(&device_count);
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
	cudaError_t err;
	CALLJ(cudaMemGetInfo(&free, &total), memory_error);
	if (free < bytes * 4 / 3)
		return false;

	return true;
memory_error:
	exit(EXIT_FAILURE);
}

bool cuda_align(struct input in, struct output out)
{
	if (no_cuda)
		return align(in, out);

	if (!cuda_device_init())
		return false;

	if (in.max > MAX_CUDA_SEQUENCE_LENGTH) {
		perr("Sequence length exceeds CUDA Device limits");
		return false;
	}

	cudaError_t err;
	uint block_max = ({
		struct cudaDeviceProp dev_prop;
		CALLR(cudaGetDeviceProperties(&dev_prop, 0));
		pinfo("Using CUDA device: %s", dev_prop.name);
		dev_prop.maxThreadsPerBlock;
	});

	struct constants C = {
		.num = in.num,
		.gap_pen = GAP_PEN,
		.gap_open = GAP_OPN,
		.gap_ext = GAP_EXT,
	};

	memcpy(C.seq_lut, SEQ_LUT, sizeof(SEQ_LUT));
	memcpy(C.sub_mat, SUB_MAT, sizeof(SUB_MAT));

	s32 num = in.num;
	s32 sum = in.meta[num - 1].off + in.meta[num - 1].len + 1;
	size_t meta_bytes = bytesof(in.meta, num);

	CALLR(cudaMalloc((void **)&C.letters, sum));
	CALLR(cudaMalloc((void **)&C.meta, meta_bytes));
	CALLR(cudaMemcpy(C.letters, in.seqs, sum, cudaMemcpyHostToDevice));
	CALLR(cudaMemcpy(C.meta, in.meta, meta_bytes, cudaMemcpyHostToDevice));

	s32 *matrix = out.matrix;
	s64 alignments = alignments((s64)num);
	constexpr s64 batch_size = 64 << 20;

	if (!cuda_memory(bytesof(matrix, num * num))) {
		if (!cuda_memory(bytesof(matrix, alignments))) {
			if (!cuda_memory(bytesof(matrix, batch_size))) {
				perr("Not enough CUDA Device memory for alignment");
				return false;
			}
		}
		C.triangular = true;
	}

	if (out.triangular)
		C.triangular = true;

	s64 batch = 0, batch_last = 0, batch_done = 0;
	void *scores[2] = {};
	s32 active = 0;
	if (C.triangular) {
		batch = min(alignments, batch_size);
		CALLR(cudaMalloc(&scores[0], bytesof(matrix, batch)));
		CALLR(cudaMemset(scores[0], 0, bytesof(matrix, batch)));
		CALLR(cudaMalloc(&scores[1], bytesof(matrix, batch)));
		CALLR(cudaMemset(scores[1], 0, bytesof(matrix, batch)));
	} else {
		batch = alignments;
		CALLR(cudaMalloc(&*scores, bytesof(matrix, num * num)));
		CALLR(cudaMemset(*scores, 0, bytesof(matrix, num * num)));
	}

	CALLR(cudaMalloc((void **)&C.progress, sizeof(*C.progress)));
	CALLR(cudaMemset(C.progress, 0, sizeof(*C.progress)));
	CALLR(cudaMemcpyToSymbol(pC, &C, sizeof(C), 0, cudaMemcpyHostToDevice));

	const void *kernel = ALIGN->kernel;
	dim3 block = { block_max, 1, 1 };
	cudaStream_t compute, memory;
	CALLR(cudaStreamCreate(&compute));
	CALLR(cudaStreamCreate(&memory));

	bool subsequent = false, syncing = false, matrix_copied = false;
	s64 progress = 0;

	pinfol("Performing %zu pairwise alignments", (size_t)alignments);

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
				CALLR(cudaMemcpy(matrix, *scores,
						 bytesof(matrix, num * num),
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
					bytesof(matrix, n_scores),
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
						 bytesof(matrix, n_scores),
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

bool cuda_memory(size_t)
{
	return true;
}

bool cuda_align(struct input in, struct output out)
{
	return align(in, out);
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
